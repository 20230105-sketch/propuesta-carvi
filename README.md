# Replicación del artículo 17

Este repositorio replica la metodología principal del artículo:

> Özüpak, Y., Aslan, E., & Zaitsev, I. (2026). Explainable LSTM-AdamW based fault diagnosis of aircraft rotating components using airborne acoustic signals under dynamic operating conditions. *Scientific Reports, 16*, 11449. https://doi.org/10.1038/s41598-026-41889-2

## Objetivo

Clasificar cuatro condiciones de un rodamiento mediante sonido:

1. Funcionamiento normal.
2. Falla en la pista interna.
3. Falla en la pista externa.
4. Falla en el elemento rodante.

## Metodología replicada

- Dataset público **Engine Acoustic Emissions** de Kaggle.
- Audio mono a 44.1 kHz.
- Segmentos de 1,000 muestras sin superposición.
- Normalización de cada segmento.
- Comparación de RNN, GRU y LSTM.
- LSTM entrenada con AdamW.
- Tamaño de lote de 32 y máximo de 50 épocas.
- Early stopping con paciencia de 10 épocas.
- Evaluación con exactitud, precisión, recall, macro-F1 y matriz de confusión.

El artículo no publica de forma explícita todos los valores necesarios para reconstruir exactamente la arquitectura, como el número de unidades ocultas y la tasa de aprendizaje. Esta réplica declara esos valores en el código (64 unidades recurrentes y tasa de aprendizaje de 0.001) para que el experimento sea transparente y reproducible. Por ello, se replica el procedimiento metodológico, pero no se garantiza obtener exactamente el 99.26 % informado por los autores.

## Diferencia importante respecto al artículo

El artículo indica cuatro grabaciones de 12 segundos a 44.1 kHz, pero el archivo descargado de Kaggle contiene un archivo `dataset.mat` con cuatro vectores de **120,000 muestras**, uno por clase. Además, el texto señala que separó grabaciones completas entre entrenamiento y prueba. Con una sola señal por clase, esa separación dejaría clases fuera de uno de los conjuntos.

Para evitar mezclar ventanas vecinas al azar, esta réplica divide cada señal cronológicamente en 70 % para entrenamiento, 15 % para validación y 15 % para prueba. Después crea ventanas de 1,000 muestras. Es una validación más exigente, aunque todavía preliminar porque cada clase procede de una única señal continua.

## Disponibilidad comprobada

- **Datos:** sí están disponibles públicamente en Kaggle: `julienjta/engine-acoustic-emissions`.
- **Contenido descargado:** `dataset.mat`, con las variables `normal`, `inner`, `outer` y `roller`.
- **Código de los autores:** no se encontró un repositorio público enlazado en el artículo.
- **Código de esta réplica:** `src/replicar_dataset_kaggle.py`.

## Cómo ejecutarlo en Google Colab

1. Abrir `replicacion_articulo17.ipynb` en Google Colab.
2. Ejecutar las celdas en orden.
3. El código descargará automáticamente el dataset público desde Kaggle.
4. Esperar el entrenamiento de los tres modelos.
5. Descargar la carpeta `resultados`.

También puede ejecutarse desde una terminal:

```bash
pip install -r requirements.txt
python src/replicar_dataset_kaggle.py
```

## Resultados generados

El programa crea:

```text
resultados/
├── metricas_modelos.csv
├── curvas_validacion.png
├── matriz_confusion_RNN.png
├── matriz_confusion_GRU.png
├── matriz_confusion_LSTM_AdamW.png
└── resumen_experimento.json
```

Los archivos `.keras` se generan localmente, pero se excluyen del repositorio para evitar almacenar modelos pesados. Las matrices, métricas y reportes sí se incluyen como evidencia de la ejecución.

## Resultados de la ejecución incluida

| Modelo | Exactitud | F1 macro |
|---|---:|---:|
| RNN | 54.17 % | 51.17 % |
| GRU | **80.56 %** | **80.50 %** |
| LSTM + AdamW | 50.00 % | 38.72 % |

El 99.26 % corresponde al artículo original y no debe presentarse como resultado propio. La diferencia muestra que el rendimiento publicado no se reproduce automáticamente con el archivo disponible y con una separación cronológica más estricta.

## Relación con la propuesta de Carvimsa

La réplica permite comprobar si los modelos recurrentes pueden aprender cambios temporales en una señal acústica. En esta ejecución, GRU obtuvo el mejor resultado. Posteriormente, la misma estructura puede adaptarse a grabaciones de la máquina corrugadora de Carvimsa, reemplazando las clases del dataset por condiciones normales y anomalías confirmadas por el personal técnico.

