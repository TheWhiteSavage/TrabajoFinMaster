"""
API de predicción de riesgo de incendio forestal — TFM.

Prueba de concepto de productivización: expone el modelo Random Forest
definitivo desarrollado en el TFM mediante un servicio REST que recibe
las variables predictoras de una celda-día y devuelve una puntuación
de riesgo relativa junto con una clasificación binaria basada en el
umbral de decisión determinado durante el proceso de modelado.

La puntuación devuelta corresponde a la salida de predict_proba() del
modelo y se utiliza como medida de priorización relativa. No debe
interpretarse como una probabilidad absoluta calibrada de ocurrencia
de incendio.

Ejecución local:
    uvicorn app.main:app --reload

Documentación interactiva (Swagger UI):
    http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from app.schemas import ObservacionRiesgo, PrediccionRiesgo

import json
import joblib
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# Rutas a los artefactos exportados desde 04_Model.ipynb
# ============================================================
RUTA_MODELOS = Path(__file__).resolve().parent.parent / "modelos"
RUTA_MODELO = RUTA_MODELOS / "rf_final.joblib"
RUTA_METADATA = RUTA_MODELOS / "rf_final_metadata.json"

# ============================================================
# ESTADO COMPARTIDO DE LA APLICACIÓN
# ============================================================
estado = {"modelo": None, "metadata": None,}

# ============================================================
# CICLO DE VIDA DE LA APLICACIÓN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga el modelo y la metadata una sola vez al arrancar
    el servicio.

    Esto evita deserializar el archivo .joblib en cada petición,
    lo que sería especialmente costoso debido al tamaño del
    modelo Random Forest.
    """
    # Comprobar existencia del modelo
    if not RUTA_MODELO.exists():
        raise RuntimeError(
            f"No se encuentra el modelo en: {RUTA_MODELO}. "
            "Copia rf_final.joblib, exportado desde 04_Model.ipynb, "
            "en la carpeta 'modelos/' antes de arrancar el servicio."
            )

    # Comprobar existencia de metadata
    if not RUTA_METADATA.exists():
        raise RuntimeError(
            f"No se encuentra la metadata en: {RUTA_METADATA}. "
            "Copia rf_final_metadata.json, exportado desde "
            "04_Model.ipynb, en la carpeta 'modelos/'."
            )

    # Cargar modelo
    estado["modelo"] = joblib.load(RUTA_MODELO)

    # Cargar metadata
    with open(RUTA_METADATA, encoding="utf-8") as f:
        estado["metadata"] = json.load(f)

    # Información de arranque
    print(f"Modelo cargado: {type(estado['modelo']).__name__}")
    print(f"Features esperadas: {len(estado['metadata']['features'])}")
    print(f"Umbral de decisión:{estado['metadata']['umbral_decision']:.4f}")

    yield

    # Limpieza al apagar el servicio
    estado.clear()


# ============================================================
# CONFIGURACIÓN DE FASTAPI
# ============================================================

app = FastAPI(
    title="Predicción de riesgo de incendio forestal — TFM",
    description=(
        "Prueba de concepto de productivización del modelo Random Forest "
        "definitivo desarrollado en el TFM 'Predicción de riesgo de "
        "incendio forestal en España mediante aprendizaje automático "
        "sobre reanálisis climático ERA5-Land y registros históricos EGIF'. "
        "\n\n"
        "La API recibe las variables predictoras de una celda-día y "
        "devuelve una puntuación de riesgo relativa. Esta puntuación "
        "se utiliza para priorizar observaciones, pero no debe "
        "interpretarse como una probabilidad absoluta calibrada."
    ), version="1.0.0", lifespan=lifespan)

# ENDPOINT RAÍZ
@app.get("/", tags=["Estado"])
def raiz():
    return {
        "mensaje": (
            "API de predicción de riesgo de incendio forestal. "
            "Consultar la documentación interactiva en /docs."
            )}

# ENDPOINT DE SALUD
@app.get("/health", tags=["Estado"])
def salud():
    if estado["modelo"] is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado"
            )

    metadata = estado["metadata"]

    return {
        "estado": "ok",
        "modelo": type(estado["modelo"]).__name__,
        "n_features": len(metadata["features"]),
        "umbral_decision": metadata["umbral_decision"]
        }

# ENDPOINT DE METADATA
@app.get("/metadata", tags=["Estado"])
def obtener_metadata():
    if estado["modelo"] is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible",
        )

    return estado["metadata"]

# ENDPOINT DE PREDICCIÓN
@app.post("/predict", response_model=PrediccionRiesgo, tags=["Predicción"])
def predecir(observacion: ObservacionRiesgo):
    """
    Recibe las variables predictoras de una celda-día y devuelve:
    - risk_score: puntuación de riesgo relativa obtenida mediante
      predict_proba().
    - umbral_decision: umbral utilizado para transformar la
      puntuación continua en una clasificación binaria.
    - prioridad_alta: True cuando la puntuación supera el umbral.
    La puntuación no debe interpretarse como una probabilidad
    absoluta calibrada de ocurrencia de incendio.
    """

    # Comprobar disponibilidad del modelo
    if estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    metadata = estado["metadata"]

    # Obtener features esperadas
    features_esperadas = metadata["features"]

    # Convertir la observación recibida a diccionario
    datos = observacion.model_dump()

    # Construir DataFrame respetando EXACTAMENTE el orden utilizado durante el entrenamiento.
    try:
        X = pd.DataFrame(
            [[datos[f] for f in features_esperadas]],
            columns=features_esperadas)

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=(f"Falta la variable requerida por el modelo: {e}"))

    # Generar puntuación
    proba = estado["modelo"].predict_proba(X)[:, 1][0]

    # Recuperar umbral de decisión
    umbral = metadata["umbral_decision"]

    # Clasificación binaria
    prioridad_alta = bool(proba >= umbral)

    # Construir respuesta
    return PrediccionRiesgo(
        risk_score=float(proba),
        umbral_decision=float(umbral),
        prioridad_alta=prioridad_alta,
        modelo=type(estado["modelo"]).__name__,)