from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    lower,
    when,
    to_date,
    coalesce,
    lead,
    avg
)
from pyspark.sql.window import Window


def crear_spark_session():
    spark = (
        SparkSession.builder
        .appName("Pipeline_Recesiones_BigData")
        .master("local[*]")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def leer_fred_md(spark, ruta_fred):
    fred = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(str(ruta_fred))
    )

    # FRED-MD suele traer una fila llamada "Transform:".
    # Esa fila no es dato económico, por eso se elimina.
    fred = fred.filter(col("sasdate").isNotNull())
    fred = fred.filter(lower(trim(col("sasdate"))) != "transform:")

    # Convertir fecha
    fred = fred.withColumn(
        "date",
        coalesce(
            to_date(col("sasdate"), "M/d/yyyy"),
            to_date(col("sasdate"), "MM/dd/yyyy"),
            to_date(col("sasdate"), "yyyy-MM-dd")
        )
    )

    fred = fred.filter(col("date").isNotNull())

    # Convertir todas las columnas económicas a numéricas
    columnas = fred.columns
    columnas_features = [c for c in columnas if c not in ["sasdate", "date"]]

    for c in columnas_features:
        fred = fred.withColumn(
            c,
            when(
                (trim(col(c)) == "") |
                (trim(col(c)) == ".") |
                (trim(col(c)) == "NA") |
                (trim(col(c)) == "#N/A"),
                None
            ).otherwise(col(c)).cast("double")
        )

    fred = fred.drop("sasdate")

    return fred


def leer_usrec(spark, ruta_usrec):
    usrec = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(str(ruta_usrec))
    )

    usrec = usrec.withColumn(
        "date",
        to_date(col("observation_date"), "yyyy-MM-dd")
    )

    usrec = usrec.select(
        "date",
        col("USREC").cast("int").alias("recession")
    )

    return usrec


def preparar_dataset():
    spark = crear_spark_session()

    raiz = Path(__file__).resolve().parents[2]

    ruta_fred = raiz / "data" / "raw" / "FRED_MD.csv"
    ruta_usrec = raiz / "data" / "raw" / "USREC.csv"

    salida_parquet = raiz / "data" / "processed" / "fred_recession_dataset.parquet"
    salida_csv = raiz / "data" / "processed" / "fred_recession_dataset_csv"

    ruta_fred_spark = f"file://{ruta_fred}"
    ruta_usrec_spark = f"file://{ruta_usrec}"
    salida_parquet_spark = f"file://{salida_parquet}"
    salida_csv_spark = f"file://{salida_csv}"

    print("Leyendo FRED-MD...")
    fred = leer_fred_md(spark, ruta_fred_spark)

    print("Leyendo USREC...")
    usrec = leer_usrec(spark, ruta_usrec_spark)

    print("Uniendo datasets por fecha...")
    dataset = fred.join(usrec, on="date", how="inner").orderBy("date")

    # Crear variable objetivo:
    # target_recession_3m = indica si habrá recesión dentro de 3 meses
    ventana = Window.orderBy("date")

    dataset = dataset.withColumn(
        "target_recession_3m",
        lead(col("recession"), 3).over(ventana)
    )

    dataset = dataset.filter(col("target_recession_3m").isNotNull())

    # Rellenar valores faltantes con promedio por columna
    columnas_features = [
        c for c in dataset.columns
        if c not in ["date", "recession", "target_recession_3m"]
    ]

    promedios = dataset.select(
        [avg(c).alias(c) for c in columnas_features]
    ).collect()[0].asDict()

    promedios = {
        k: v for k, v in promedios.items()
        if v is not None
    }

    dataset = dataset.fillna(promedios)

    print("Muestra del dataset final:")
    dataset.select(
        "date",
        "recession",
        "target_recession_3m"
    ).show(10)

    print("Columnas totales:", len(dataset.columns))
    print("Registros totales:", dataset.count())

    print("Guardando dataset en Parquet...")
    dataset.write.mode("overwrite").parquet(salida_parquet_spark)


    print("Guardando copia en CSV...")
    dataset.coalesce(1).write.mode("overwrite").option("header", True).csv(salida_csv_spark)
    print("Dataset preparado correctamente.")
    print(f"Archivo Parquet: {salida_parquet}")
    print(f"Carpeta CSV: {salida_csv}")

    spark.stop()


if __name__ == "__main__":
    preparar_dataset()