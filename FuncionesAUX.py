# IMPORTS
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    precision_recall_curve)

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np


# RUTAS DE TRABAJO
BASE = Path(r"C:\Users\Gambo\Documents\Master Aaron\TFM UCM")
DATA = BASE / "Data"
FIGURES = BASE / "Figures"


# CAPAS CARTOGRÁFICAS
# Se cargan una única vez al importar el módulo para reutilizarlas en todos los
# notebooks sin volver a leer el shapefile desde disco.
ruta_provincias = (
    DATA
    / "SHP_ETRS89"
    / "ll_provinciales_inspire_peninbal_etrs89"
    / "ll_provinciales_inspire_peninbal_etrs89.shp")

PROVINCIAS = gpd.read_file(ruta_provincias).to_crs(epsg=4326)


# FUNCIONES
def print_header(title, width=60):
    """
    Imprime un encabezado visual llamativo para organizar las celdas de Jupyter.
    title (str)          : Texto que se mostrará en el centro del encabezado
    width (int/opcional) : Longitud de las líneas decorativas. Por defecto es 60
    """

    print("=" * width)
    print(title)
    print("=" * width)


def save_figure(filename: str, directory: Path = FIGURES, dpi: int = 300, bbox: str = "tight",):
    """
    Guarda la figura actual en el directorio indicado.
    directory (Path) : Carpeta donde guardar la figura.
    filename  (str)  : Nombre del archivo (con o sin extensión).
    dpi       (int)  : Resolución de salida.
    bbox      (str)  : Bounding box utilizado por matplotlib.
    """

    directory.mkdir(parents=True, exist_ok=True)

    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(".png")

    plt.tight_layout()
    plt.savefig(directory / path, dpi=dpi, bbox_inches=bbox)
    plt.show()



def plot_provincias(ax, linewidth=0.5, alpha=0.6, color="black", zorder=2):
    """
    Dibuja los límites provinciales sobre un eje de Matplotlib.
    """
    PROVINCIAS.plot(
        ax=ax,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder)



def calcular_class_weight(y):
    """
    Calcula los pesos de clase equivalentes a class_weight='balanced' de sklearn.
    Devuelve:
        class_weight      -> dict para sklearn
        scale_pos_weight  -> float para XGBoost / LightGBM
    """

    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    if n_pos == 0:
        raise ValueError("El conjunto no contiene positivos.")

    w_pos = n / (2 * n_pos)
    w_neg = n / (2 * n_neg)

    return (
        {0: w_neg, 1: w_pos},
        w_pos / w_neg
    )

def construir_cv_walkforward(df, folds):
    """Devuelve una lista de (train_idx, val_idx) en posiciones
    enteras (iloc-compatibles), a partir de la definición de folds ya existente."""
    assert df.index.equals(pd.RangeIndex(len(df))), (
    "El DataFrame debe tener un índice consecutivo. "
    "Use reset_index(drop=True) antes de construir los folds.")
    cv_splits = []
    for f in folds:
        train_idx = df.index[df["anio"].isin(f["train_years"])].to_numpy()
        val_idx   = df.index[df["anio"] == f["val_year"]].to_numpy()
        cv_splits.append((train_idx, val_idx))
    return cv_splits


# ============================================================
# CONSTRUCCIÓN DE MATRICES X/y
# ============================================================
def construir_matrices(df, features, target):
    """
    Devuelve matrices X e y listas para entrenamiento.
    """
    X = df[features].copy()
    y = df[target].copy()
    assert X.isnull().sum().sum() == 0, "NaN en X"
    assert y.isnull().sum() == 0, "NaN en y"
    return X, y

def obtener_fold(df, fold, features, target):
    """
    Construye X_train, y_train, X_val, y_val para un fold walk-forward.
    """
    train = df[df["anio"].isin(fold["train_years"])]
    val   = df[df["anio"] == fold["val_year"]]
    X_train, y_train = construir_matrices(train, features, target)
    X_val, y_val     = construir_matrices(val, features, target)
    class_weight, scale_pos_weight = calcular_class_weight(y_train)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "class_weight": class_weight, "scale_pos_weight": scale_pos_weight,
    }

def obtener_holdout(df, train_years, test_year, features, target):
    """
    Construye las matrices del entrenamiento final y del holdout,
    filtrando por años directamente sobre `df` — misma mecánica
    que obtener_fold, para no mantener train_final/test_final
    como DataFrames construidos por separado en otra celda.
    """
    train = df[df["anio"].isin(train_years)]
    test  = df[df["anio"] == test_year]
    X_train, y_train = construir_matrices(train, features, target)
    X_test, y_test   = construir_matrices(test, features, target)
    class_weight, scale_pos_weight = calcular_class_weight(y_train)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "class_weight": class_weight, "scale_pos_weight": scale_pos_weight,
    }


# MODELOS: Random Forest - XGBoost - LightGBM
# ============================================================
# MÉTRICAS
# ============================================================

def evaluar_modelo(y_true, y_proba, threshold=0.5):
    """
    Evalúa un clasificador binario.
    Parametros
        y_true : array
        y_proba : probabilidades clase positiva
        threshold : float
    Returns
        dict
    """

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "PR-AUC": average_precision_score(y_true, y_proba),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(y_true, y_pred),
        "Accuracy (informativa)": accuracy_score(
            y_true,
            y_pred
        )
    }

# ============================================================
# SELECCIÓN DE UMBRAL (sin fuga)
# ============================================================
def obtener_umbral_optimo(y_true, y_proba):
    """
    Devuelve el umbral que maximiza F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = (2 * precision * recall / (precision + recall + 1e-10))
    idx = np.argmax(f1[:-1])
    return thresholds[idx]

# ============================================================
# RESUMEN WALK-FORWARD
# ============================================================
def resumir_resultados(resultados):
    df = pd.DataFrame(resultados)
    print(df.round(4))
    print("\nResumen")
    for m in [
        "PR-AUC",
        "ROC-AUC",
        "Recall",
        "Precision",
        "F1",
    ]:
        print(
            f"{m:10s}: "
            f"{df[m].mean():.4f} ± "
            f"{df[m].std():.4f}"
        )
    return df

# ============================================================
# ENTRENAMIENTO FINAL
# ============================================================

def entrenar_modelo_final(modelo, X_train, y_train, X_test):
    """
    Entrena el modelo definitivo y devuelve probabilidades del test.
    NOTA: modelo debe llegar ya configurado con el class_weight/
    scale_pos_weight decidido (ver bloque de configuración por modelo)
    — esta función no lo aplica ni lo verifica.
    """
    modelo.fit(X_train, y_train)
    return modelo.predict_proba(X_test)[:, 1]


# ============================================================
# FUNCIÓN — RECALL@TOP-K% CON INTERVALO DE CONFIANZA
#
# Selecciona el K% de observaciones con mayor probabilidad
# predicha y calcula qué proporción de los incendios reales
# queda capturada dentro de ese grupo.
#
# El IC95% se obtiene mediante bootstrap sobre las observaciones.
# ============================================================

def recall_en_top_k_bootstrap(proba, y_true, porcentaje, n_bootstrap=1000, random_state=42):
    """
    Calcula Recall@Top-K% y su intervalo de confianza bootstrap.

    Parámetros
    proba : array-like      Probabilidades predichas de la clase positiva.
    y_true : array-like     Etiquetas reales (0/1).
    porcentaje : float      Porcentaje de observaciones que se seleccionan,
    n_bootstrap : int       Número de réplicas bootstrap.
    random_state : int      Semilla reproducible.

    Retorna
    ic_inf : float        Límite inferior del IC95%.
    mediana : float       Mediana del recall bootstrap.
    ic_sup : float        Límite superior del IC95%.
    """

    proba = np.asarray(proba)
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    k = max(1, int(np.ceil(n * porcentaje / 100)))

    recalls = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        proba_b = proba[indices]
        y_b = y_true[indices]

        # Número total de positivos en la muestra bootstrap
        n_positivos = y_b.sum()
        if n_positivos == 0:
            continue

        # Seleccionar Top-K%
        indices_top = np.argsort(proba_b)[-k:]
        positivos_detectados = y_b[indices_top].sum()
        recall = positivos_detectados / n_positivos
        recalls.append(recall)

    recalls = np.asarray(recalls)

    ic_inf = np.percentile(recalls, 2.5)
    mediana = np.percentile(recalls, 50)
    ic_sup = np.percentile(recalls, 97.5)

    return ic_inf, mediana, ic_sup
