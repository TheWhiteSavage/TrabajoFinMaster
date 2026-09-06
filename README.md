# API de predicción de riesgo de incendio forestal — TFM
Prueba de concepto de productivización del modelo Random Forest definitivo desarrollado en el marco del Trabajo de Fin de Máster.

## Requisitos e Instalación
1. Clonar o descargar este repositorio en tu entorno local (VSCode / terminal).
2. Crear y activar un entorno virtual de Python:
   ```bash
   # En Linux / macOS:
   python -m venv venv
   source venv/bin/activate

   # En Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Instalar las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

## Configuración de Artefactos del Modelo
Para el correcto funcionamiento del servicio con el modelo definitivo, es necesario situar los artefactos generados durante la fase de modelado dentro del directorio modelos/. 

Asegúrate de que la carpeta contenga los siguientes archivos con sus nombres exactos:
- modelos/rf_final.joblib (Binario del modelo entrenado)
- modelos/rf_final_metadata.json (Metadatos del modelo: variables y umbral óptimo)

> **Nota:** Por defecto, el repositorio incluye artefactos mínimos de prueba (dummy) diseñados exclusivamente para validar el despliegue técnico de la arquitectura de la API. Deben ser sustituidos por los definitivos para obtener predicciones reales.

## Ejecución del Servicio
Levanta el servidor de desarrollo local mediante Uvicorn:

```bash
uvicorn app.main:app --reload
```

Una vez iniciado, la documentación interactiva (Swagger UI) estará disponible de forma automática en:  
http://127.0.0.1:8000/docs

## Endpoints Principales
El servicio expone una interfaz REST estructurada en los siguientes puntos de acceso:

- GET / — Mensaje de bienvenida e instrucciones de uso.
- GET /health — Verificación del estado del servicio y confirmación de carga del modelo.
- GET /metadata — Consulta de las variables requeridas y el umbral de decisión configurado.
- POST /predict — Recibe las 18 variables predictoras de una celda-día y devuelve la puntuación de riesgo relativa, el umbral de decisión y la clasificación de prioridad alta.

## Alcance y Limitaciones
Este servicio web está diseñado para procesar variables predictoras previamente calculadas y estructuradas según el esquema de datos. La ingesta automatizada, la sincronización y la ingeniería de variables en tiempo real a partir de las fuentes primarias (reanálisis climático ERA5-Land y registros históricos de incendios) quedan fuera del alcance de esta prueba de concepto, planteándose como una línea de desarrollo futuro dentro de la memoria del TFM.