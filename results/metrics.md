# Resultados del modelo

Métricas calculadas sobre el set de prueba (10 000 casos: 5 000 imágenes con
arma, 5 000 sin arma), tomadas de la evaluación final de la tesis
(`valores_matriz_confusion1.xlsx`, enero 2025).

## Matriz de confusión

| | Predice **Arma** | Predice **Sin arma** | Total real |
|---|---:|---:|---:|
| **Con arma** (real) | TP = 4712 | FN = 288 | 5000 |
| **Sin arma** (real) | FP = 453 | TN = 4547 | 5000 |

## Métricas agregadas

| Métrica | Valor |
|---|---:|
| Precisión (Precision) | 91.23 % |
| Recall (Sensibilidad) | 94.24 % |
| Exactitud (Accuracy) | 92.59 % |

## Curva de entrenamiento (Darknet)

Entrenado por 10 000 iteraciones (`max_batches=10000`). El mAP subió de
forma consistente pasando 40 % → 89 %, estabilizándose entre 85-89 % desde
la iteración ~8000; el loss promedio bajó de forma monótona hasta ~0.15-0.03.
Ver las imágenes en [`training_curves/`](training_curves/).

## Limitaciones observadas

Revisando manualmente las salidas en [`samples/`](samples/), el modelo:

- Detecta bien armas de fuego reales en distintas poses y fondos, incluso
  parcialmente ocultas sobre una mesa (ver `resultado_05.jpg`).
- Generaliza a réplicas/juguetes con forma de arma (ver `resultado_20.jpg`,
  una pistola de luz azul de arcade correctamente marcada como "Arma").
- Puede generar **falsos positivos** sobre objetos con formas o siluetas
  parecidas (botellas, herramientas) cuando aparecen junto a un arma real en
  la misma escena, aunque con confianza notablemente más baja que las
  detecciones correctas — de ahí la importancia de usar un umbral de
  confianza razonable (`--confianza 0.5` por defecto) en vez del `> 0` de
  los scripts originales de la tesis.
