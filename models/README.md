# Modelos

Los archivos `.cfg` de este repo (`yolov3_custom.cfg`,
`configuracion_deteccion_armas.cfg`) sí están incluidos: son texto plano y
pesan unos KB. Los pesos entrenados (`.weights`, ~235 MB) **no** se subieron
al repositorio porque superan el límite de 100 MB de GitHub.

## Cómo conseguir los pesos

El modelo final usado en la tesis es `yolov3_custom_last_13122024.weights`
(YOLOv3, una sola clase `Arma`, entrenado con Darknet sobre un dataset propio
de imágenes de armas de fuego).

Opciones para tenerlo disponible localmente:

1. **Git LFS** (recomendado si vas a versionar los pesos): agrega el `.weights`
   con `git lfs track "*.weights"` y súbelo a un repo con LFS habilitado.
2. **Release de GitHub**: sube el `.weights` como asset adjunto de un
   [Release](https://docs.github.com/en/repositories/releasing-projects-on-github)
   en vez de comitearlo — no cuenta contra el límite de tamaño del repo.
3. **Almacenamiento externo** (Google Drive, Hugging Face Hub, etc.) y enlázalo
   aquí.

Una vez que tengas el archivo, colócalo en esta carpeta (`models/`) con el
mismo nombre que usan los scripts por defecto, o pasa la ruta explícita con
`--pesos`:

```bash
python scripts/detectar_imagen.py --imagen foto.jpg \
    --pesos models/yolov3_custom_last_13122024.weights \
    --config models/yolov3_custom.cfg
```

## Resultados del modelo entrenado

Ver [`results/metrics.md`](../results/metrics.md) para precisión, recall y
matriz de confusión sobre el set de prueba, y
[`results/training_curves/`](../results/training_curves/) para las curvas de
loss y mAP durante el entrenamiento en Darknet.
