"""
Esquemas de validación de entrada y salida para la API de predicción de riesgo.

La API requiere la recepción de las 18 variables predictoras previamente
calculadas y consolidadas durante la fase de modelado del TFM.

En un entorno de producción maduro, un pipeline de ingeniería de datos previo 
se encargaría de extraer, transformar y derivar automáticamente estas variables 
a partir de las fuentes primarias (ERA5-Land, MDT, EGIF/CIVIO) para una coordenada 
y ventana temporal específicas.
"""

from pydantic import BaseModel, Field

class ObservacionRiesgo(BaseModel):
    """Variables predictoras para una celda-día concreta."""
    
    # Meteorología — ERA5-Land
    swvl1: float = Field(..., description="Humedad del suelo, capa 0-7 cm")
    swvl2: float = Field(..., description="Humedad del suelo, capa 7-28 cm")
    t2m: float = Field(..., description="Temperatura a 2 m (Kelvin)")
    tp: float = Field(..., description="Precipitación total diaria (m)")

    # Soil Water Index (SWI)
    swi_T10_swvl1: float = Field(..., description="SWI con T=10 aplicado a swvl1")
    swi_T10_swvl2: float = Field(..., description="SWI con T=10 aplicado a swvl2")
    swi_T20_swvl1: float = Field(..., description="SWI con T=20 aplicado a swvl1")
    swi_T20_swvl2: float = Field(..., description="SWI con T=20 aplicado a swvl2")
    swi_T40_swvl1: float = Field(..., description="SWI con T=40 aplicado a swvl1")
    swi_T40_swvl2: float = Field(..., description="SWI con T=40 aplicado a swvl2")

    # Topografía — SRTM
    elevacion: float = Field(..., description="Elevación media (m)")
    pendiente: float = Field(..., description="Pendiente media (grados)")

    # Presión histórica de ignición — EGIF/CIVIO
    densidad_igniciones: float = Field(..., description="Número histórico de incendios en la celda")
    pct_causa_antropica: float = Field(..., ge=0, le=1, description="Proporción histórica de causas antrópicas")
    pct_causa_rayo: float = Field(..., ge=0, le=1, description="Proporción histórica de causas por rayo")

    # Variables temporales
    dias_desde_inicio: float = Field(
        ..., 
        description="Días transcurridos desde el hito temporal de referencia (2008-01-01)."
    )
    mes_sin: float = Field(..., ge=-1, le=1, description="Componente armónico seno del día juliano")
    mes_cos: float = Field(..., ge=-1, le=1, description="Componente armónico coseno del día juliano")

    # Ejemplo unificado para Swagger UI
    model_config = {
        "json_schema_extra": {
            "example": {
                "swvl1": 0.18,
                "swvl2": 0.21,
                "t2m": 298.4,
                "tp": 0.0,
                "swi_T10_swvl1": 0.22,
                "swi_T10_swvl2": 0.24,
                "swi_T20_swvl1": 0.25,
                "swi_T20_swvl2": 0.27,
                "swi_T40_swvl1": 0.28,
                "swi_T40_swvl2": 0.30,
                "elevacion": 450.0,
                "pendiente": 8.5,
                "densidad_igniciones": 95.0,
                "pct_causa_antropica": 0.72,
                "pct_causa_rayo": 0.05,
                "dias_desde_inicio": 5000.0,
                "mes_sin": 0.5,
                "mes_cos": -0.5
            }
        }
    }


class PrediccionRiesgo(BaseModel):
    """Respuesta del endpoint de predicción."""

    risk_score: float = Field(
        ...,
        description="Puntuación de riesgo relativa obtenida mediante predict_proba(). No representa una probabilidad absoluta calibrada."
    )
    umbral_decision: float = Field(
        ...,
        description="Umbral óptimo de decisión determinado estadísticamente durante el modelado."
    )
    prioridad_alta: bool = Field(
        ...,
        description="Indicador binario. True si el risk_score iguala o supera el umbral establecido."
    )
    modelo: str = Field(..., description="Identificador de la arquitectura del modelo utilizado.")
