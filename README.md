# proyecto_BigData

## Proyecto

¿Es posible detectar o anticipar periodos de recesión económica a partir de indicadores macroeconómicos históricos usando herramientas de Big Data?

## Descripción

Este proyecto desarrolla un pipeline de Big Data para procesar indicadores macroeconómicos históricos y analizar su relación con periodos de recesión económica.

## Fuentes de datos

- FRED-MD: indicadores macroeconómicos mensuales.
- USREC: indicador binario de recesión económica en Estados Unidos.

## Arquitectura general

1. Ingesta de datos macroeconómicos.
2. Limpieza y transformación con PySpark.
3. Almacenamiento en formato procesado.
4. Entrenamiento de modelo predictivo con SparkML.
5. Visualización de resultados mediante dashboard.

## Estructura del repositorio

```text
data/
notebooks/
src/
reports/
docs/

```

## Tecnologías

- Python
- PySpark
- SparkML
- Git y GitHub

## Ejecución

```bash
pip install -r requirements.txt
python main.py
```

## Herramientas

- Python
- PySpark
- SparkML
- Git/GitHub
- Visual Studio Code
- Dashboard con Streamlit o Power BI
