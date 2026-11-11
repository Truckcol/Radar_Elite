import joblib
import pandas as pd
import numpy as np

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 10)


def preprocesar_y_predecir(nuevos_jugadores_df, ruta_paquete='models/modelo_scouting_rf.joblib'):
    # 1. Cargar el paquete serializado
    paquete = joblib.load(ruta_paquete)
    modelo = paquete['modelo']
    columnas_entrenamiento = paquete['columnas_X']

    df = nuevos_jugadores_df.copy()

    # 2. Aplicar las MISMAS reglas de imputación aprendidas en entrenamiento
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    df['estatura_cm'] = df['estatura_cm'].fillna(df['posicion'].map(paquete['medianas_estatura']))
    df['estatura_cm'] = df['estatura_cm'].fillna(paquete['mediana_global_estatura'])
    df['edad'] = df['edad'].fillna(paquete['mediana_global_edad'])

    # 3. Formatear variables categóricas alineadas con la matriz original
    # Sin drop_first: en un lote pequeño la primera categoría presente no siempre es
    # 'Attack' (la referencia del entrenamiento); el reindex descarta la columna sobrante
    df_dummies = pd.get_dummies(df[['posicion']], dtype=int)
    X_nuevo = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)

    # Reindexar para asegurar que contenga exactamente las mismas columnas que vio el modelo
    X_nuevo = X_nuevo.reindex(columns=columnas_entrenamiento, fill_value=0)

    # 4. Generar inferencias y probabilidades
    predicciones = modelo.predict(X_nuevo)
    probabilidades = modelo.predict_proba(X_nuevo)[:, 1]

    df['Prediccion_Clase'] = np.where(predicciones == 1, 'Élite', 'Estándar')
    df['Probabilidad_Elite'] = np.round(probabilidades * 100, 2)

    return df


def main():
    print("=================================================================")
    print("  FASE 2: SIMULACIÓN DE INFERENCIA EN PRODUCCIÓN (NUEVOS DATOS)")
    print("=================================================================\n")

    # Simulación de datos que llegan desde el frontend o una API externa
    nuevos_candidatos = pd.DataFrame([
        {'jugador': 'Candidato A', 'posicion': 'Defender', 'estatura_cm': 188, 'edad': 22},
        {'jugador': 'Candidato B', 'posicion': 'Goalkeeper', 'estatura_cm': None, 'edad': 29},
        {'jugador': 'Candidato C', 'posicion': 'Midfield', 'estatura_cm': 172, 'edad': 19}
    ])

    print("Entrada de datos crudos:")
    print(nuevos_candidatos)
    print("\nProcesando inferencia con el modelo serializado...")

    resultados = preprocesar_y_predecir(nuevos_candidatos)

    print("\n=================================================")
    print("  RESULTADO DEL DIAGNÓSTICO DE SCOUTING")
    print("=================================================")
    print(resultados[['jugador', 'posicion', 'edad', 'estatura_cm', 'Prediccion_Clase', 'Probabilidad_Elite']])
    print("=================================================\n")


if __name__ == "__main__":
    main()
