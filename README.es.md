# Wan2GP WanFlow UI

## Capturas

### Nodo adaptado al modelo

Cada nodo de generacion muestra los puertos y controles que admite el modelo seleccionado. El ejemplo muestra un nodo de video con entradas de prompt, imagen, video, audio, mascara, frame final y frames inyectados.

![Nodo Generate Video](assets/screenshots/generate-video-node.png)

### Workflow conectado y grupo reutilizable

El canvas permite ramas conectadas, previews, mascaras, postprocesado y grupos reutilizables que se pueden activar o desactivar como una unidad.

![Canvas de WanFlow UI](assets/screenshots/workflow-canvas.png)

### Previews rapidas con nodos y grupos desactivados

Los nodos y grupos reutilizables se pueden desactivar sin borrarlos. Asi es posible saltear una mascara, una generacion o una etapa de postprocesado para hacer una preview rapida y volver a activarla para el render final.

![Nodo desactivado](assets/screenshots/disabled-node.png)

![Desactivar un grupo reutilizable](assets/screenshots/disable-group.gif)

WanFlow UI es un editor visual de workflows por nodos para Wan2GP. Ofrece un canvas inspirado en ComfyUI y conserva los contratos nativos de Wan2GP para modelos, procesadores, medios, cola y FFmpeg.

Es un plugin de Wan2GP. No es un motor de inferencia independiente ni reemplaza la administración de modelos o el runtime de Wan2GP.

## Características

- Nodos movibles, puertos tipados, conexiones SVG, zoom, desplazamiento y ajuste de vista.
- Ejecución topológica con ramas y convergencias.
- Generación, edición, inpainting, referencias, máscaras, audio, LoRAs, resolución, aspect ratio y controles nativos por modelo.
- Análisis IA con salidas de texto y variables del workflow.
- Postprocesadores descubiertos desde el catálogo instalado de Wan2GP.
- Operaciones FFmpeg tipadas usando los binarios administrados por Wan2GP.
- Previews de imágenes, videos, audio y máscaras.
- Grupos de color reutilizables con activación individual por nodo o por grupo.
- Bloques reutilizables guardados como subgrafos editables.

## Requisitos

- Una instalación compatible de Wan2GP.
- El entorno Python y Gradio provisto por Wan2GP.
- El catálogo de modelos y postprocesadores instalados en Wan2GP.
- Los binarios `ffmpeg` y `ffprobe` administrados por Wan2GP.

No se necesitan librerías JavaScript externas. Los modelos, LoRAs, procesadores y archivos multimedia grandes no están incluidos.

## Instalación

Copia la carpeta `wan2gp-wanflow-ui` dentro de la carpeta `plugins` de Wan2GP, reinicia Wan2GP y abre la pestaña `WanFlow UI`.

## Uso básico

1. Agrega nodos desde la paleta.
2. Arrastra desde un puerto de salida hasta una entrada compatible.
3. Selecciona un nodo para editarlo en el Inspector.
4. Carga medios runtime y asigna los nodos de entrada a sus slots.
5. Guarda el workflow o pulsa `Run`.

El runner valida puertos, entradas obligatorias, capacidades del modelo y ciclos antes de ejecutar.

El nodo `Magic Mask` acepta una imagen o un video y palabras clave como `person, car, sky`. Conecta `mask_image` a un nodo de inpainting de imagen o `mask_video` a uno de video. Usa los recursos administrados por Wan2GP y los descarga la primera vez que hacen falta.

Los nodos de generación también aceptan URLs remotas de LoRA. Agrégalas desde la sección LoRAs del Inspector y configura su fuerza; Wan2GP resuelve y guarda la URL en caché al ejecutar el workflow.

## Grupos y bloques

Selecciona varios nodos con `Shift` y usa `+ Group` o `Ctrl+G`. Los grupos pueden renombrarse, colorearse, moverse y activarse o desactivarse. Un grupo de postprocesado simple puede desactivarse para obtener una preview rápida.

Desde el Inspector del grupo, `Save as reusable block` guarda un subgrafo editable que luego aparece en la paleta.

## Desarrollo

```powershell
venv\Scripts\python.exe -m unittest discover -s plugins/wan2gp-wanflow-ui/tests -v
node --check plugins/wan2gp-wanflow-ui/assets/editor.js
```

## Licencia

El código original de WanFlow UI incluido en este repositorio está bajo la Apache License 2.0; consulta [`LICENSE`](LICENSE).

WanFlow UI es un complemento para WanGP/Wan2GP y no relicencia WanGP/Wan2GP. El texto de la licencia aplicable de WanGP/Wan2GP se conserva en [`LICENSE.txt`](LICENSE.txt). Al redistribuir WanGP/Wan2GP junto con este plugin, conserva esa licencia y los avisos correspondientes.

Los modelos, LoRAs, procesadores, FFmpeg, paquetes de Python y demás materiales de terceros no quedan relicenciados por este proyecto. Mantienen sus propios términos. Consulta [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
