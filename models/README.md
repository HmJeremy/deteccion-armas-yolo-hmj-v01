# Modelos

Los `.cfg` de este repo (`yolov3_custom.cfg`,
`configuracion_deteccion_armas.cfg`) sí están acá, son texto plano y pesan
un par de KB nomás. Los pesos entrenados (`.weights`, ~235 MB) **no** los
subí al repositorio porque se pasan del límite de 100 MB que pone GitHub.

## Cómo conseguir los pesos

El modelo final que usé en la tesis es
`yolov3_custom_last_13122024.weights` (YOLOv3, una sola clase `Arma`,
entrenado con Darknet sobre mi dataset de fotos de armas de fuego).

Para tenerlo disponible en tu máquina tienes un par de opciones:

1. **Git LFS** (la más práctica si vas a versionar los pesos): agrega el
   `.weights` con `git lfs track "*.weights"` y súbelo a un repo con LFS
   habilitado.
2. **Release de GitHub**: sube el `.weights` como asset adjunto de un
   [Release](https://docs.github.com/en/repositories/releasing-projects-on-github)
   en vez de comitearlo — así no cuenta contra el límite de tamaño del repo.
3. **Almacenamiento externo** (Google Drive, Hugging Face Hub, lo que
   tengas a mano) y lo enlazas acá.

Una vez que tengas el archivo, ponlo en esta carpeta (`models/`) con el
mismo nombre que usan los scripts por defecto, o pásale la ruta directa con
`--pesos`:

```bash
python scripts/detectar_imagen.py --imagen foto.jpg \
    --pesos models/yolov3_custom_last_13122024.weights \
    --config models/yolov3_custom.cfg
```

## Resultados del modelo entrenado

En [`results/metrics.md`](../results/metrics.md) están la precisión, el
recall y la matriz de confusión sobre el set de prueba, y en
[`results/training_curves/`](../results/training_curves/) las curvas de
loss y mAP del entrenamiento en Darknet.
