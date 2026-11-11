import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

RUTA_MODELO = 'models/modelo_scouting_rf.joblib'

# Se carga el paquete serializado una sola vez al iniciar el servidor
paquete = joblib.load(RUTA_MODELO)
modelo = paquete['modelo']
columnas_entrenamiento = paquete['columnas_X']

app = FastAPI(
    title="API de Scouting Deportivo",
    description="Predice si un jugador pertenece al segmento Élite (Top 25% de valor comercial) "
                "usando un RandomForestClassifier serializado con Joblib.",
    version="1.0.0",
)

# Permite que el aplicativo web (index.html) consuma la API desde otro origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Jugador(BaseModel):
    """Esquema de entrada validado por Pydantic"""
    jugador: str = Field("Jugador Nuevo", examples=["Candidato A"])
    posicion: str = Field(..., examples=["Defender"])
    edad: Optional[float] = Field(None, ge=14, le=50, examples=[22])
    estatura_cm: Optional[float] = Field(None, ge=140, le=220, examples=[188])


class Prediccion(BaseModel):
    """Esquema de salida de la API"""
    jugador: str
    posicion: str
    edad: float
    estatura_cm: float
    prediccion: int
    prediccion_clase: str
    probabilidad_elite: float
    umbral_elite_eur: float


def preprocesar(jugador: Jugador) -> pd.DataFrame:
    df = pd.DataFrame([jugador.model_dump()])

    # Mismas reglas de imputación aprendidas en el entrenamiento
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['estatura_cm'] = df['estatura_cm'].fillna(df['posicion'].map(paquete['medianas_estatura']))
    df['estatura_cm'] = df['estatura_cm'].fillna(paquete['mediana_global_estatura'])
    df['edad'] = df['edad'].fillna(paquete['mediana_global_edad'])
    return df


@app.get("/")
def inicio():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": type(modelo).__name__,
        "posiciones_validas": paquete['posiciones_validas'],
        "auc_test": round(paquete['auc_test'], 4),
    }


@app.post("/predict", response_model=Prediccion)
def predict(jugador: Jugador):
    if jugador.posicion not in paquete['posiciones_validas']:
        raise HTTPException(
            status_code=422,
            detail=f"Posición inválida. Use una de: {paquete['posiciones_validas']}",
        )

    df = preprocesar(jugador)

    # One-Hot Encoding alineado con las columnas que vio el modelo
    df_dummies = pd.get_dummies(df[['posicion']], dtype=int)
    X_nuevo = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    X_nuevo = X_nuevo.reindex(columns=columnas_entrenamiento, fill_value=0)

    prediccion = int(modelo.predict(X_nuevo)[0])
    probabilidad = float(modelo.predict_proba(X_nuevo)[0, 1])

    return Prediccion(
        jugador=jugador.jugador,
        posicion=jugador.posicion,
        edad=float(df['edad'].iloc[0]),
        estatura_cm=float(df['estatura_cm'].iloc[0]),
        prediccion=prediccion,
        prediccion_clase="Élite" if prediccion == 1 else "Estándar",
        probabilidad_elite=round(probabilidad * 100, 2),
        umbral_elite_eur=paquete['umbral_elite_eur'],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=False)
