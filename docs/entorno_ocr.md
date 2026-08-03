# Entorno de ejecución del OCR — Docling + SuryaOCR

Guía del entorno en el que corre la etapa 0 del pipeline: cómo se construye,
por qué hacen falta parches de compatibilidad, y qué hacer cuando algo falla.

## Arquitectura

```
 Host (fteran)
 ┌──────────────────────────────────────────────────────────┐
 │ runner/datahub_ctl.sh    ← control (run/ensure/stop/...) │
 │ runner/crontab.datahub   ← @reboot + recreate 22:00      │
 │ deploy.sh                ← despliegue completo           │
 │ Dockerfile               ← imagen reproducible           │
 │ instalar_ocr.sh          ← instalación del entorno       │
 │ requirements_ocr.txt     ← versiones verificadas         │
 │ docling_suryaOCR.py      ← script OCR (con parches)      │
 │ runner/supervisor.sh     ← PID 1, un proceso por GPU     │
 │ runner/balance_folders.py← reparto LPT por pendientes    │
 │ runner/progress.py       ← tabla de progreso             │
 │ venvs/surya2/            ← se monta en /opt/venv         │
 └──────────────────────────────────────────────────────────┘
                    │
                    ▼
 Contenedor datahub_ocr
 ┌──────────────────────────────────────┐
 │ supervisor.sh (PID 1)                │
 │   ├─ GPU lógica 0 → docling_suryaOCR │
 │   └─ GPU lógica 1 → docling_suryaOCR │
 └──────────────────────────────────────┘
```

Docker expone solo las GPU físicas indicadas en `DEVICES` (por defecto 2,3) y
las renumera como 0 y 1 dentro del contenedor. Por eso `GPUS="0 1"` en
`datahub_ctl.sh` es correcto y no debe tocarse.

---

## Construcción del entorno

### 1. Imagen Docker

```bash
docker build -t datahub-ocr:latest -f Dockerfile .
```

La imagen parte de `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` y delega
toda la instalación en `instalar_ocr.sh`.

**Requisitos que fijan esa base:** GLIBC ≥ 2.38 y Python 3.12, necesarios para
los wheels modernos. Ubuntu 24.04 trae GLIBC 2.39 y Python 3.12; **Ubuntu
22.04 no sirve** (GLIBC 2.35, Python 3.10), aunque versiones anteriores de esta
guía lo indicaban.

> **Pendiente de verificar.** El entorno que produjo el corpus se creó de forma
> interactiva y se congeló con `docker commit`, no con este Dockerfile. El
> Dockerfile reconstruye el mismo procedimiento pero **no se ha validado con un
> build completo**. Antes de confiar en él, hay que ejecutarlo y comprobar la
> salida de verificación de `instalar_ocr.sh`.

### 2. Virtualenv

El venv se crea **dentro del contenedor**, para heredar su GLIBC, y se monta en
él en tiempo de ejecución. Eso permite cambiar dependencias sin reconstruir la
imagen.

```bash
docker run --rm -it \
  -v /home/fteran/Proyecto_DataHub/venvs:/venvs \
  -v /home/fteran/Proyecto_DataHub/instalar_ocr.sh:/tmp/instalar_ocr.sh:ro \
  datahub-ocr:latest bash

# Dentro del contenedor:
python3.12 -m venv /venvs/surya2
/venvs/surya2/bin/pip install --upgrade pip setuptools wheel
bash /tmp/instalar_ocr.sh /venvs/surya2
```

### 3. Instalación de dependencias

Todo el procedimiento vive en **`instalar_ocr.sh`**, que es la única fuente de
verdad: lo usan tanto el Dockerfile como la creación del venv. No repitas los
comandos aquí ni en otro sitio, o acabarán divergiendo la próxima vez que
cambie una versión.

El script instala en cuatro fases y el **orden es obligatorio**:

1. PyTorch desde el índice CUDA 12.8 (no desde PyPI).
2. Dependencias base: pillow, transformers, pydantic, accelerate.
3. `surya-ocr`, `docling-surya` y `docling` con `--no-deps`.
4. Las dependencias de docling que el paso anterior dejó fuera.

**Por qué `--no-deps`:** surya-ocr y docling declaran dependencias de torch. Si
pip las resuelve, sustituye la build de CUDA por una de CPU. El OCR sigue
funcionando, pero unas ochenta veces más lento y **sin ningún error visible**.

El script termina verificando que torch vea CUDA y aborta si no. Las versiones
exactas están en `requirements_ocr.txt`, que sirve de registro y no para
instalar directamente.

---

## Parches de compatibilidad

`surya-ocr 0.17.1` se escribió contra una API de `transformers` anterior a la
4.51. Los parches en `docling_suryaOCR.py` la reconstruyen:

| # | Síntoma | Causa raíz | Solución |
|---|---------|-----------|----------|
| 1 | `AttributeError: pad_token_id` | `SuryaDecoderConfig` no tiene el atributo | `SuryaDecoderConfig.pad_token_id = None` |
| 2 | `KeyError: 'default'` en `ROPE_INIT_FUNCTIONS` | transformers 4.51+ quitó la clave `'default'` | Función propia que usa `d.get("factor", 1.0)` |
| 3 | `AttributeError: all_tied_weights_keys` | transformers 4.51+ lo fija en `post_init()`, que surya nunca llama | Asignarlo en `__init__` |
| 4 | `tie_weights() got unexpected keyword argument` | transformers pasa `missing_keys`, `recompute_mapping` | Envoltorio con `**kwargs` |
| 5 | `AttributeError: _tie_or_clone_weights` | Método eliminado de transformers | Reimplementación propia |
| 6 | Tensores en *meta device* en el encoder de visión | `inv_freq` se crea como tensor plano, no como buffer | Recrearlo en CPU dentro del `forward` |

Son la parte más frágil del pipeline: si uno falla, el síntoma aparece mucho
después disfrazado de error del modelo. Por eso **el script reporta al arrancar
cuáles se aplicaron**:

```
[parches] 6/6 aplicados
```

Si alguno falla, imprime cuál y con qué error, y sigue adelante. Ante cualquier
fallo raro del OCR, esa línea es lo primero que hay que mirar.

---

## Despliegue y operación

```bash
./deploy.sh                          # despliegue completo (interactivo)
./runner/datahub_ctl.sh run          # arrancar
./runner/datahub_ctl.sh status       # estado + tabla de progreso
./runner/datahub_ctl.sh logs         # salida en vivo
crontab runner/crontab.datahub       # programación (una sola vez)
./runner/datahub_ctl.sh stop         # detener
```

El contenedor tiene tres redes de reinicio superpuestas: `supervisor.sh`
relanza el proceso caído, `--restart=always` relanza el contenedor, y `@reboot`
lo levanta tras un arranque de la máquina. **Consecuencia:** la única forma de
pararlo de verdad es `datahub_ctl.sh stop` *y* quitar el crontab. Si solo matas
el proceso, vuelve.

Consulta del avance sin detener nada:

```bash
kill -USR1 <pid>                     # el pid lo imprime el script al arrancar
python3 runner/progress.py --watch 30
```

---

## Troubleshooting

**El OCR corre pero en CPU.** El síntoma es el ritmo: decenas de veces más
lento de lo normal. Casi siempre es pip habiendo reemplazado torch por la build
de CPU. Verifica con `python -c "import torch; print(torch.version.cuda,
torch.cuda.is_available())"` y reinstala con `instalar_ocr.sh`.

**`venvs/surya2/bin/python: No such file or directory` desde el host.** No está
dañado: el venv se creó dentro del contenedor y su intérprete es un enlace a un
Python que solo existe ahí. Úsalo desde dentro del contenedor.

**El factory dice "No class found" pero suryaocr aparece listado.** Falso
negativo: `create_instance()` en `base_factory.py` captura `KeyError` tanto
para el lookup como para errores internos del modelo. Revisa el log completo,
no solo el último mensaje.

**Errores de GLIBC al crear el venv.** Se está creando en el host en vez de
dentro del contenedor. Ver la sección 2.

**CUDA out of memory.** Baja `PARALLEL` en `datahub_ctl.sh` (por defecto 4).
Surya usa unos 2-3 GB por proceso de OCR.

**Los modelos no se descargan.** Comprueba `HOME=/opt/cache` entre las
variables del contenedor y que `$CACHE_HOST` tenga espacio libre.

**El contenedor revive solo después de pararlo.** Es `--restart=always` o el
`@reboot` del crontab. Ver "Despliegue y operación".

**Imágenes que no se procesan.** El script solo toma las extensiones de
`--ext` (por defecto jpg, jpeg, png, tif, tiff). Al arrancar imprime el
desglose por extensión; si el conteo no cuadra con lo que esperas, ahí está la
pista.