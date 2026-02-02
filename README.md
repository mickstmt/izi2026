# Izistore 2026 Financial Projections Dashboard

Este proyecto es una aplicación web de análisis de datos construida con Flask, Pandas y Plotly para proyectar el flujo de caja y la rentabilidad de Izistore en 2026 basándose en datos históricos.

## Características

- **ETL Automático:** Ingesta y limpieza de datos desde Excel.
- **Motor de Proyecciones:** Cálculos basados en ratios históricos y crecimiento esperado.
- **Dashboard Interactivo:** Visualizaciones dinámicas con Plotly y Bootstrap.
- **Análisis de Escenarios:** Recalcula proyecciones ajustando variables de crecimiento y marketing.

## Estructura del Proyecto

- `app.py`: Servidor Flask y rutas.
- `engine/data_processor.py`: Lógica de transformación de datos y proyecciones.
- `templates/`: Plantillas HTML.
- `static/`: Archivos CSS y JS.
- `data/`: Directorio para el archivo Excel de origen.

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`
