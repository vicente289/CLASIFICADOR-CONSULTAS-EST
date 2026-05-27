# Clasificador Automático de Consultas Estudiantiles

Proyecto académico de Inteligencia Artificial orientado a clasificar automáticamente consultas estudiantiles escritas en español.

## Área y subárea

- Área: Aprendizaje Supervisado
- Subárea: Procesamiento del Lenguaje Natural
- Problema: Clasificación automática de texto en múltiples categorías académicas

## Objetivo

Crear una aplicación funcional donde el usuario escriba una consulta y el sistema devuelva la categoría más probable usando un modelo clásico de Inteligencia Artificial.

## Tecnologías usadas

- Python
- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib
- seaborn
- joblib
- re
- pytest

## Categorías

1. Inscripciones
2. Horarios
3. Pagos
4. Notas
5. Trámites
6. Plataforma virtual
7. Becas
8. Exámenes
9. Otros

## Estructura del proyecto

```text
clasificador_consultas_estudiantiles/
├── app.py
├── train_model.py
├── dataset_consultas.csv
├── modelo_consultas.pkl
├── vectorizador_tfidf.pkl
├── metrics_report.txt
├── confusion_matrix.png
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── predict.py
│   └── evaluate.py
└── tests/
    └── test_basic.py
```

## Instalación

Desde la carpeta del proyecto:

```bash
pip install -r requirements.txt
```

## Entrenar el modelo

```bash
python train_model.py
```

Este comando:

- carga `dataset_consultas.csv`
- limpia las consultas
- divide los datos en 80% entrenamiento y 20% prueba
- vectoriza el texto con TF-IDF
- entrena Regresión Logística Multiclase
- entrena Multinomial Naive Bayes como comparación
- selecciona el modelo con mejor F1-score macro
- genera `modelo_consultas.pkl`
- genera `vectorizador_tfidf.pkl`
- genera `metrics_report.txt`
- genera `confusion_matrix.png`

## Ejecutar la aplicación

```bash
streamlit run app.py
```

Luego abre la URL local que muestra Streamlit, normalmente:

```text
http://localhost:8501
```

## Ejemplos de consultas

- No puedo ingresar a la plataforma
- ¿Cuánto debo pagar?
- ¿Cuándo es el parcial?
- Necesito una constancia
- ¿Dónde veo mi horario?
- Quiero renovar mi beca
- No aparece mi nota final
- Hasta cuándo puedo inscribirme

## Explicación de TF-IDF

TF-IDF significa Term Frequency - Inverse Document Frequency. Es una técnica para convertir texto en números. Da más peso a las palabras importantes de una consulta y reduce el peso de palabras muy repetidas en todo el dataset. En este proyecto permite representar cada consulta como un vector que puede ser usado por modelos de clasificación.

## Explicación de Regresión Logística Multiclase

La Regresión Logística Multiclase es un modelo supervisado que aprende a separar categorías a partir de ejemplos etiquetados. Aunque su nombre menciona regresión, se usa ampliamente para clasificación. En este proyecto recibe vectores TF-IDF y predice una de las nueve categorías.

## Modelo comparativo

También se entrena Multinomial Naive Bayes. Este modelo es común en clasificación de texto porque trabaja bien con conteos o pesos de palabras. Se compara con Regresión Logística usando accuracy y F1-score macro.

## Métricas

- Accuracy: proporción total de predicciones correctas.
- Precision: mide cuántas predicciones de una categoría fueron correctas.
- Recall: mide cuántos casos reales de una categoría fueron detectados.
- F1-score: combina precision y recall.
- F1-score macro: promedio del F1-score de todas las categorías, útil cuando se quiere evaluar el rendimiento general por clase.
- Matriz de confusión: muestra aciertos y errores por categoría real y predicha.

## Pruebas básicas

```bash
pytest
```

Las pruebas verifican que el dataset exista, tenga las columnas correctas, incluya las nueve categorías y que la limpieza de texto devuelva un resultado válido.

## Limitaciones del prototipo

- El dataset es académico y sintético, no proviene de un sistema real de atención estudiantil.
- Puede fallar con consultas muy ambiguas o con vocabulario no visto en entrenamiento.
- No guarda historial de consultas.
- No usa modelos profundos ni embeddings contextuales, porque el objetivo es mantener un enfoque clásico y defendible.

## Mejoras futuras

- Ampliar el dataset con consultas reales anonimizadas.
- Agregar validación cruzada.
- Incorporar balanceo y análisis de errores por categoría.
- Permitir retroalimentación del usuario para corregir predicciones.
- Crear un panel administrativo para revisar consultas frecuentes.
- Integrar el clasificador con un sistema de atención estudiantil.
