from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def crear_spark_session():
    spark = (
        SparkSession.builder
        .appName("Modelo_Recesiones_BigData")
        .master("local[*]")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def cargar_dataset(spark, ruta_dataset):
    dataset = spark.read.parquet(f"file://{ruta_dataset}")
    return dataset


def dividir_dataset_temporal(dataset):
    """
    División temporal:
    - 80% inicial para entrenamiento
    - 20% final para prueba

    Esto es mejor que una división aleatoria porque los datos son series de tiempo.
    """

    ventana = Window.orderBy("date")

    dataset = dataset.withColumn("row_id", row_number().over(ventana))

    total = dataset.count()
    limite_train = int(total * 0.8)

    train = dataset.filter(col("row_id") <= limite_train)
    test = dataset.filter(col("row_id") > limite_train)

    train = train.drop("row_id")
    test = test.drop("row_id")

    return train, test


def evaluar_modelo(predicciones, nombre_modelo):
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="target_recession_3m",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="accuracy"
    )

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="f1"
    )

    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="target_recession_3m",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    auc = evaluator_auc.evaluate(predicciones)
    accuracy = evaluator_accuracy.evaluate(predicciones)
    f1 = evaluator_f1.evaluate(predicciones)
    precision = evaluator_precision.evaluate(predicciones)
    recall = evaluator_recall.evaluate(predicciones)

    print("\n====================================")
    print(f"Resultados del modelo: {nombre_modelo}")
    print("====================================")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

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

    columnas_excluir = ["date", "recession", "target_recession_3m"]
    columnas_features = [
        c for c in dataset.columns
        if c not in columnas_excluir
    ]

    print("Variables usadas para entrenar:", len(columnas_features))

    train, test = dividir_dataset_temporal(dataset)

    print("Registros de entrenamiento:", train.count())
    print("Registros de prueba:", test.count())

    assembler = VectorAssembler(
        inputCols=columnas_features,
        outputCol="features_raw",
        handleInvalid="keep"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    rf = RandomForestClassifier(
        labelCol="target_recession_3m",
        featuresCol="features",
        numTrees=100,
        maxDepth=6,
        seed=42
    )

    lr = LogisticRegression(
        labelCol="target_recession_3m",
        featuresCol="features",
        maxIter=50
    )

    pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
    pipeline_lr = Pipeline(stages=[assembler, scaler, lr])

    print("\nEntrenando Random Forest...")
    modelo_rf = pipeline_rf.fit(train)
    predicciones_rf = modelo_rf.transform(test)

    resultados_rf = evaluar_modelo(predicciones_rf, "Random Forest")

    print("\nEntrenando Logistic Regression...")
    modelo_lr = pipeline_lr.fit(train)
    predicciones_lr = modelo_lr.transform(test)

    resultados_lr = evaluar_modelo(predicciones_lr, "Logistic Regression")

    print("\nGuardando modelos...")
    modelo_rf.write().overwrite().save(f"file://{ruta_modelo_rf}")
    modelo_lr.write().overwrite().save(f"file://{ruta_modelo_lr}")

    print("Guardando predicciones del mejor modelo en CSV...")
    predicciones_rf_export =(
        predicciones_rf
        .withColumn("prob_no_recession", vector_to_array("probability")[0])
        .withColumn("prob_recession", vector_to_array("probability")[1])
    )
    
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

    spark.stop()


if __name__ == "__main__":
    entrenar_modelos()