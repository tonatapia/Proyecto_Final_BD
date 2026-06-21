from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


#En esta primer función inicializa y devuelve una sesión de spark configurada para ejecucipon local,
#usa todos los nucleos sisponibles con (local[*]) y desactiva logs de nivel
#INFO/WARN para mantener la salida limpia durante el entrenamiento.
def crear_spark_session():
    spark = (
        SparkSession.builder
        .appName("Modelo_Recesiones_BigData")
        .master("local[*]")
        #usa todos los nucleos del sistema
        .config("spark.hadoop.fs.defaultFS", "file:///")
        #sistema de archivos local en vez de HDFS
        .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR") #suprime logs INFO y WARN
    return spark


#para esta segunda función se lee el dataser en formato parquet desda la ruta indicada y lo devuelve como un dataframe de Spark.
def cargar_dataset(spark, ruta_dataset):
    dataset = spark.read.parquet(f"file://{ruta_dataset}")
    return dataset




#Tercer función, este enfoque evita fuga fuga de información futura hacia el conjunto de entrenamiento lo cual queda y es importante en series de tiempo económicas.
def dividir_dataset_temporal(dataset):
    """
    División temporal:
    - 80% inicial para entrenamiento
    - 20% final para prueba
    Esto es mejor que una división aleatoria porque los datos son series de tiempo.
    """

    #ventana ordenada por fecha para asignar un indice secuencial a cada fila
    ventana = Window.orderBy("date")

    dataset = dataset.withColumn("row_id", row_number().over(ventana))

    total = dataset.count()
    limite_train = int(total * 0.8) #ocurre el corte del 80 en orden cronológico.
    
    train = dataset.filter(col("row_id") <= limite_train)
    test = dataset.filter(col("row_id") > limite_train)
    
    #elimina columna auxiliar antes de devolver.
    train = train.drop("row_id")
    test = test.drop("row_id")

    return train, test


#cuarta función, calcula e imprime las métricas de clasificación binaria para un modelo entrenado e incluye la matriz de confusión agrupada por clase real y clase predicha.
def evaluar_modelo(predicciones, nombre_modelo):
    #definición de evaluadores por metrica
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="target_recession_3m",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC" #AUC-ROC que tan bien separa clases
    )

    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="accuracy"  #proporción de predicciones correctas.
    )

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="f1" #media armonica de precisión y recall
    )

    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="weightedPrecision" #precisión ponderada por soporte de clase
    )

    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="weightedRecall" #recall ponderado por soporte de clase
    )
    
    #calculo de metricas
    auc = evaluator_auc.evaluate(predicciones)
    accuracy = evaluator_accuracy.evaluate(predicciones)
    f1 = evaluator_f1.evaluate(predicciones)
    precision = evaluator_precision.evaluate(predicciones)
    recall = evaluator_recall.evaluate(predicciones)
    
    #impresión de resultados 
    print("\n====================================")
    print(f"Resultados del modelo: {nombre_modelo}")
    print("====================================")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    #matriz de confusión: filas=clase real, clumnas=clase predicha
    print("\nMatriz de confusión:")
    predicciones.groupBy("target_recession_3m", "prediction").count().orderBy(
        "target_recession_3m", "prediction"
    ).show()

    return {
        "modelo": nombre_modelo,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


#quinta función, es la función prinicpal del pipeline de entrenamiento.
#se realizan las siguientes etapas: 
#1. inicialización de spark, 2.carga del dataset procesado en parquet, 3.division 80train/20test
#4. Construcción de pipelines de preprocesamiento + clasificación, 5.Entrenamiento y evaluación de Random Forest y Regresión Logística
def entrenar_modelos():
    spark = crear_spark_session()

    raiz = Path(__file__).resolve().parents[2]

    ruta_dataset = raiz / "data" / "processed" / "fred_recession_dataset.parquet"
    ruta_modelo_rf = raiz / "models" / "random_forest_recesion"
    ruta_modelo_lr = raiz / "models" / "logistic_regression_recesion"
    ruta_predicciones = raiz / "results" / "predicciones_recesion"

    print("Cargando dataset procesado...")
    dataset = cargar_dataset(spark, ruta_dataset)

    print("Columnas disponibles:", len(dataset.columns))
    print("Registros disponibles:", dataset.count())
    
    #excluye la etiqueta objetivo y columnas de identificación temporal
    columnas_excluir = ["date", "recession", "target_recession_3m"]
    columnas_features = [
        c for c in dataset.columns
        if c not in columnas_excluir
    ]

    print("Variables usadas para entrenar:", len(columnas_features))

    train, test = dividir_dataset_temporal(dataset)

    print("Registros de entrenamiento:", train.count())
    print("Registros de prueba:", test.count())
    
    #etapa 1: combina feautures en un vecror denso
    assembler = VectorAssembler(
        inputCols=columnas_features,
        outputCol="features_raw",
        handleInvalid="keep" #conserva filas con nulos en lugar de eliminarlas.
    )

    #etapa 2:  estandarizar escala (media=0, std=1 por columna)
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False #withMean=False evita matrices densas en datos dispersos
    )

    #Definicion de clasificadores
    rf = RandomForestClassifier(
        labelCol="target_recession_3m",
        featuresCol="features",
        numTrees=100, #número de árboles en el ensamble
        maxDepth=6,  #prifundidad maxima para controlar sobreajuste
        seed=42     #semilla de reproducibilidad
    )

    lr = LogisticRegression(
        labelCol="target_recession_3m",
        featuresCol="features",
        maxIter=50 #iteraciones máximas del optimizador L-BFGS
    )

    #Construcción de pipeline: preprocesamiento + modelo
    pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
    pipeline_lr = Pipeline(stages=[assembler, scaler, lr])

    #entrenamiento y evaluación de Random Forest
    print("\nEntrenando Random Forest...")
    modelo_rf = pipeline_rf.fit(train)
    predicciones_rf = modelo_rf.transform(test)

    resultados_rf = evaluar_modelo(predicciones_rf, "Random Forest")

    #entrenamiento y evaluación de Regresión Logística
    print("\nEntrenando Logistic Regression...")
    modelo_lr = pipeline_lr.fit(train)
    predicciones_lr = modelo_lr.transform(test)
    resultados_lr = evaluar_modelo(predicciones_lr, "Logistic Regression")

    #persistencia de modelos en formato nativo de Spark ML
    print("\nGuardando modelos...")
    modelo_rf.write().overwrite().save(f"file://{ruta_modelo_rf}")
    modelo_lr.write().overwrite().save(f"file://{ruta_modelo_lr}")

    #exportar predicciones del RF con probabilidades desempaquetadas del vector
    print("Guardando predicciones del mejor modelo en CSV...")
    predicciones_rf_export =(
        predicciones_rf
        .withColumn("prob_no_recession", vector_to_array("probability")[0]) # P(clase=0)
        .withColumn("prob_recession", vector_to_array("probability")[1]) # P(clase=1)
    )

    # coalesce(1) unifica las particiones en un solo archivo CSV
    predicciones_rf_export.select(
        "date",
        "recession",
        "target_recession_3m",
        "prediction",
        "prob_no_recession",
        "prob_recession"
    ).coalesce(1).write.mode("overwrite").option("header", True).csv(
    f"file://{ruta_predicciones}"
    )

    print("\nProceso de entrenamiento terminado.")
    print(f"Modelo Random Forest guardado en: {ruta_modelo_rf}")
    print(f"Modelo Logistic Regression guardado en: {ruta_modelo_lr}")
    print(f"Predicciones guardadas en: {ruta_predicciones}")

    spark.stop() #libera recursos de Spark al finalizar.


if __name__ == "__main__":
    entrenar_modelos()