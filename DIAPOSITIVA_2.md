# Diapositiva 2. Experimentación y prototipo

## Datos

- Engine Acoustic Emissions, disponible en Kaggle.
- Cuatro condiciones: normal, pista interna, pista externa y elemento rodante.
- El archivo descargado contiene cuatro señales de 120,000 muestras.

## Preparación

- División cronológica 70/15/15 antes de segmentar.
- Segmentos de 1,000 muestras.
- Ventanas sin superposición.
- Normalización de cada segmento.

## Modelos comparados

- RNN.
- GRU.
- LSTM optimizada con AdamW.

## Resultados propios

- Exactitud del mejor modelo: **80.56 %**.
- Macro-F1: **80.50 %**.
- Modelo seleccionado: **GRU**.
- El artículo reporta 99.26 % para LSTM.

## Código

Colocar aquí el enlace del repositorio:

El enlace se agrega después de publicar este repositorio.

## Explicación oral breve

“Replicamos la metodología del artículo con el dataset público Engine Acoustic Emissions. Dividimos cronológicamente las señales antes de crear fragmentos de 1,000 muestras y comparamos RNN, GRU y LSTM. El mejor modelo de nuestra ejecución fue GRU, con 80.56 % de exactitud, mientras que el artículo reportó 99.26 % para LSTM. La diferencia puede relacionarse con la división de los datos y con hiperparámetros que el artículo no detalla.”

