import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Carga las variables definidas en el archivo .env
load_dotenv()


class MySQLConnector:
    def __init__(self):
        # Lectura de credenciales desde el entorno
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME", "")

        # Cadena de conexión parametrizada
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
        self.engine = create_engine(self.connection_string)

    def extract_data(self, query: str) -> pd.DataFrame:
        """Ejecuta la consulta y retorna un DataFrame de Pandas"""
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection)
                print(f"[OK] Extracción exitosa. Registros obtenidos: {len(df)}")
                return df
        except Exception as e:
            print(f"[ERROR] Error al consultar MySQL: {e}")
            raise
