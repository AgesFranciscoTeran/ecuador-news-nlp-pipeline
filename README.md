# Corpus de prensa ecuatoriana — pipeline de procesamiento

Pipeline que convierte ~800 000 páginas de periódico escaneado en un corpus
unificado con fecha auditable y en una base de datos vectorial consultable por
significado y por fecha.

Cubre El Comercio, El Telégrafo y El Universo entre 2001 y 2022 (prensa
impresa, vía OCR) y Diario Expreso, El Universo y Primicias entre 2014 y 2026
(edición digital, vía webscraping).

> **Este repositorio contiene solo el código.** El corpus, la base vectorial y
> la salida del OCR viven en el servidor y en un respaldo en disco; ver
> "Dónde están los datos".

## Resultados

| Métrica | Valor |
|---|---|
| Archivos producidos por el OCR | 1 508 139 |
| Filas del corpus maestro | 9 077 527 |
| Fragmentos en la base vectorial | 9 484 560 |
| Cobertura de días fechados | 66.8% / 82.8% / 93.2% (Comercio / Telégrafo / Universo) |
| Recuperación con ventana temporal | 100% de recall en 59 ms |
| Recuperación global | 98.3% de recall en 383 ms |

---

## Etapa 0 — OCR

`docling_suryaOCR.py` convierte cada imagen escaneada en dos archivos: `raw.md`
(el texto) y `docling.json` (el texto **más** su estructura: cajas,
coordenadas, etiquetas semánticas, orden de lectura). Se conservan ambos: el
JSON es la única vía para re-derivar el texto en el futuro —por ejemplo, para
corregir el orden de lectura en páginas a varias columnas— sin repetir semanas
de OCR.

Es una corrida de días, así que el diseño gira en torno a sobrevivir:

- **Idempotencia.** Toda imagen con `raw.md` se salta. Sobre esa propiedad se
  construye todo lo demás.
- **Tolerancia a fallos.** Cada error se escribe en un `ERROR.txt` junto a la
  imagen y el proceso sigue; una página corrupta no tumba la corrida.
- **Triple red de reinicio.** `supervisor.sh` relanza el proceso caído,
  `--restart=always` relanza el contenedor, y `@reboot` en cron lo levanta tras
  un arranque de la máquina.
- **Reparto balanceado.** `balance_folders.py` reparte el trabajo entre GPU
  según las imágenes *pendientes* (LPT a nivel de mes), y rebalancea en cada
  arranque.
- **Observabilidad.** `kill -USR1 <pid>` imprime el avance sin detener nada;
  `progress.py` da la tabla de progreso.

### Arquitectura de ejecución

```
Host                                   Contenedor datahub_ocr
├─ datahub_ctl.sh   run/stop/status    └─ supervisor.sh (PID 1)
├─ crontab.datahub  @reboot + 22:00       ├─ GPU 0 → docling_suryaOCR.py
├─ deploy.sh        despliegue            └─ GPU 1 → docling_suryaOCR.py
└─ venvs/surya2/ → montado en /opt/venv
```

> **Sobre `venvs/surya2`:** se crea *dentro* del contenedor, para heredar su
> GLIBC, y se monta en él en tiempo de ejecución. Visto desde el host su
> `bin/python` aparece como enlace roto — no está dañado, simplemente apunta a
> un intérprete que solo existe dentro de la imagen.

Los detalles del entorno, la tabla de parches de compatibilidad entre
surya-ocr y transformers, y el troubleshooting están en
[`docs/entorno_ocr.md`](docs/entorno_ocr.md).

---

## Etapas 1-8 — del OCR a la base vectorial

### 1. Fechado — `corpus_to_csv.py`

Segmenta cada página en artículos por sus encabezados y asigna fecha a
**todas** las filas mediante una cascada de estrategias, cada una etiquetada en
la columna `fecha_fuente`:

| Etiqueta | Cómo se obtuvo la fecha | Confianza |
|---|---|---|
| `header` | Leída del encabezado de la página | Día exacto |
| `pagina_anterior` / `retro` | Heredada del ancla del mismo día | Día exacto |
| `cuerpo` | Rescatada del cuerpo, validada con el día de la semana | Día exacto |
| `cuerpo_ant` / `cuerpo_retro` | Heredada de un ancla de cuerpo | Día probable |
| `bloque_mes` / `bloque_quincena` | Solo se conoce el mes o la quincena | Agregados mensuales |

El día de la semana funciona como suma de verificación: si el encabezado dice
"martes 15 de mayo del 2007" se comprueba contra el calendario, y una fecha mal
leída solo pasa la prueba con probabilidad 1/7.

### 2. Limpieza — `limpiar_corpus.py`

Elimina artefactos del OCR (tablas markdown mal convertidas, etiquetas HTML
residuales, guiones de corte de línea, confusión 0/1 dentro de palabras,
capitulares perdidas, mobiliario de portada) y **marca sin borrar** las filas
que no son artículos periodísticos: `clasificado`, `numerico`, `tabla`,
`solo_titulo`. El filtrado queda como decisión explícita de quien use el
corpus.

### 3. Auditoría — `auditar_calidad.py`

Compara el corpus del OCR contra el del webscraping como patrón de referencia,
separando **calidad de extracción** de **composición del contenido**.

### 4. Cobertura — `cobertura_temporal.py`

Mapa de qué días existen por fuente y periódico, con huecos y solapes.

### 5. Empalme — `empalmar_csv.py`

Unifica ambas fuentes y deduplica por artículo en los días de solape,
comparando titulares normalizados dentro del mismo (periódico, fecha).

Esquema del corpus maestro:

```
id, origen, id_original, periodico, fecha, fecha_fuente,
seccion, titulo, texto, flags, path
```

### 6. Base vectorial — `construir_vectordb.py`

Trocea los artículos, los embebe con `intfloat/multilingual-e5-large` (1024
dimensiones) y los indexa en LanceDB. Antepone el título a cada fragmento y
mantiene un manifiesto por (periódico, origen, mes) que permite **reindexar
solo los meses nuevos** en cada actualización.

### 7. Medición del recall — `verificar_recall.py`

Compara el índice aproximado contra la búsqueda exacta y barre los parámetros.
Reveló que la configuración por defecto del motor daba 51.7% de recall, lo que
llevó a reconstruir el índice con compresión escalar y a añadir un índice
B-tree sobre la fecha.

### 8. Consulta — `consultar_vectordb.py`

Aplica la política medida: con filtro de fechas usa búsqueda exacta (100% de
recall); sin filtro, el índice con `nprobes=128 refine=20`. Colapsa por
artículo para que el top-k sean noticias distintas.

```bash
python3 consultar_vectordb.py --db vectordb \
  --q "crisis económica y precio del petróleo" \
  --desde 2008-09-15 --hasta 2008-09-30 --k 10
```

---

## Utilidades

- `vistazo_maestro.py` — inspecciona el corpus sin abrirlo: fichas de filas,
  búsqueda por `id`/`path`/texto, y censo de composición.
- `senales_diarias.py` — panel diario de señales para el proyecto de nowcasting
  del PIB.
- `empaquetar.sh` / `verificar_paquetes.sh` — empaquetado por diario y año de la
  salida del OCR, con verificación de integridad y checksums.
- `respaldo.sh` — armado del respaldo en disco externo.

## Instalación

El entorno del OCR **no** se instala con un `pip install -r` plano: surya-ocr y
docling declaran dependencias de torch que, si pip las resuelve, sustituyen la
build de CUDA por una de CPU sin dar ningún error. El orden correcto está en un
solo lugar:

```bash
./instalar_ocr.sh /ruta/al/venv      # o sin argumento, en el Python actual
docker build -t datahub-ocr:latest -f Dockerfile .   # usa el mismo script
```

El entorno requiere GLIBC ≥ 2.38 y Python 3.12, de ahí la base `ubuntu24.04`
del Dockerfile. El procedimiento está reconstruido a partir del entorno real,
pero el build completo aún no se ha validado: ver la nota en
[`docs/entorno_ocr.md`](docs/entorno_ocr.md).

Las etapas 1-8 sí se instalan normalmente:

```bash
pip install -r requirements_pipeline.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

Pasa `--device cuda` explícito al construir la base vectorial: sin él, si CUDA
no está disponible el proceso cae a CPU en silencio y tarda unas ochenta veces
más.

## Dónde están los datos

No están en este repositorio por tamaño. Las imágenes de entrada viven fuera
del proyecto, en `/home/fteran/dhub`. El respaldo en disco externo contiene:

| Carpeta | Contenido | Tamaño |
|---|---|---|
| `ocr_output/` | 64 archivos `.tar.gz` con la salida del OCR | 11 GB |
| `corpus_maestro/` | Corpus unificado | 3.4 GB |
| `vectordb/` | Base vectorial LanceDB | 56 GB |
| `modelo/` | Snapshot del modelo de embeddings | 2.2 GB |

El modelo se conserva porque generar vectores con otra versión los volvería
incomparables con los ya indexados, y ese fallo no produce ningún error
visible: solo peores resultados.