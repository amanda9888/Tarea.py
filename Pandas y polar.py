import pandas as pd
import polars as pl

# Crear un DataFrame de ejemplo con Pandas
data_pandas = {
    'Nombre': ['Alice', 'Bob', 'Charlie'],
    'Edad': [25, 30, 35],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia']
}
df_pandas = pd.DataFrame(data_pandas)

# Mostrar el DataFrame de Pandas
print("DataFrame de Pandas:")
print(df_pandas)

# Crear un DataFrame de ejemplo con Polars
data_polars = {
    'Nombre': ['Alice', 'Bob', 'Charlie'],
    'Edad': [25, 30, 35],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia']
}
df_polars = pl.DataFrame(data_polars)

# Mostrar el DataFrame de Polars
print("\nDataFrame de Polars:")
print(df_polars)
