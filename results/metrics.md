# Resultados del modelo

Estas métricas salen del set de prueba (10 000 casos: 5 000 imágenes con
arma, 5 000 sin arma), de la evaluación final de la tesis
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

Entrené por 10 000 iteraciones (`max_batches=10000`). El mAP fue subiendo
de forma consistente, de 40 % a 89 %, y se estabilizó entre 85-89 % desde
más o menos la iteración 8000; el loss promedio bajó parejo hasta quedarse
en algo como 0.15-0.03. Las imágenes de la curva están en
[`training_curves/`](training_curves/) si quieres verlas.

## Cosas que noté revisando los resultados

Revisando a mano las salidas en [`samples/`](samples/), esto es lo que vi:

- Detecta bien las armas de fuego reales en distintas poses y fondos,
  incluso cuando están medio ocultas sobre una mesa (mira
  `resultado_05.jpg`).
- También generaliza a réplicas/juguetes con forma de arma —
  `resultado_20.jpg` es una pistola de luz azul de esas de arcade, y la
  marcó bien como "Arma".
- Sí genera algunos **falsos positivos** con objetos que tienen formas o
  siluetas parecidas (botellas, herramientas), sobre todo cuando aparecen
  junto a un arma real en la misma escena. Eso sí, con una confianza bastante
  más baja que las detecciones correctas, así que se nota que un umbral
  razonable (`--confianza 0.5` por defecto) ayuda un montón frente al `> 0`
  que usaban los scripts originales de la tesis.
