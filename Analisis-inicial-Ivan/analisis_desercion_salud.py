"""Analisis exploratorio y modelado de desercion para ClientesCDPSalud.

Este script:
1. Lee las tres bases CSV de la carpeta ClientesCDPSalud.
2. Hace un resumen exploratorio de cada base.
3. Agrega las tablas largas a nivel de ID_ficticio.
4. Construye un set unificado por cliente.
5. Propone dos enfoques de modelado para desercion.
6. Incluye un analisis exploratorio de variables latentes con Factor Analysis.

La variable objetivo se construye con comportamiento futuro de pagos/compras:
si un cliente no realiza movimientos de tipo Abono o Compra en los
siguientes N meses (por defecto 3), se marca como desercion.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42
MAX_MODEL_ROWS = 200_000
CHUNK_SIZE = 500_000
DEFAULT_CHURN_HORIZON_MONTHS = 3
DEFAULT_RENEWAL_GRACE_AFTER_DAYS = 45


def _resolve_data_dir(base_dir: Path) -> Path:
    local = base_dir / "ClientesCDPSalud"
    if local.exists():
        return local
    parent = base_dir.parent / "ClientesCDPSalud"
    if parent.exists():
        return parent
    return local


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = _resolve_data_dir(BASE_DIR)
OUTPUT_DIR = BASE_DIR / "outputs_desercion"

CUSTOMER_FILE = DATA_DIR / "Clientes_CDPsalud.csv"
SOCIO_FILE = DATA_DIR / "sociodemograficos_clientes_salud.csv"
TRANS_FILE = DATA_DIR / "transacciones_clientes_salud.csv"


def _clean_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def read_sample(path: Path, nrows: int = 50_000) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    if "" in df.columns:
        df = df.drop(columns=[""])
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = _clean_string_series(df[col])
    return df


def print_eda(name: str, df: pd.DataFrame, max_levels: int = 8) -> None:
    print("=" * 100)
    print(name)
    print("shape:", df.shape)
    print("missing ratio (top 12):")
    missing = df.isna().mean().sort_values(ascending=False)
    print(missing.head(12).to_string())
    cat_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    if cat_cols:
        print("categorical highlights:")
        for col in cat_cols[:5]:
            print(f"- {col}")
            print(df[col].value_counts(dropna=False).head(max_levels).to_string())


def _accumulate_count_table(
    acc: pd.DataFrame | None, new: pd.DataFrame
) -> pd.DataFrame:
    new = new.fillna(0)
    if acc is None:
        return new
    return acc.add(new, fill_value=0)


def _accumulate_min_series(acc: pd.Series | None, new: pd.Series) -> pd.Series:
    if acc is None:
        return new
    merged = pd.concat([acc, new], axis=1)
    return merged.min(axis=1)


def _accumulate_max_series(acc: pd.Series | None, new: pd.Series) -> pd.Series:
    if acc is None:
        return new
    merged = pd.concat([acc, new], axis=1)
    return merged.max(axis=1)


def aggregate_client_activity(
    path: Path,
    chunksize: int = CHUNK_SIZE,
    reference_dates: pd.Series | None = None,
) -> pd.DataFrame:
    usecols = [
        "ID_ficticio",
        "fechacorte_seguro",
        "FechaMovimiento",
        "FechaVencimiento",
        "Movimiento",
        "Plan",
        "Canal",
        "CantidadMesesPagados",
    ]

    count_tables: dict[str, pd.DataFrame | None] = {
        "movimiento": None,
        "plan": None,
        "canal": None,
    }
    numeric_sums: pd.DataFrame | None = None
    min_move_date: pd.Series | None = None
    max_move_date: pd.Series | None = None
    max_cutoff_date: pd.Series | None = None
    max_due_date: pd.Series | None = None

    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        chunk = chunk.loc[:, ~chunk.columns.astype(str).str.match(r"^Unnamed")]
        chunk["CantidadMesesPagados"] = pd.to_numeric(
            chunk["CantidadMesesPagados"], errors="coerce"
        )
        for col in ["fechacorte_seguro", "FechaMovimiento", "FechaVencimiento"]:
            chunk[col] = pd.to_datetime(chunk[col], errors="coerce")
        if reference_dates is not None:
            ref = chunk["ID_ficticio"].map(reference_dates)
            chunk = chunk[ref.notna() & chunk["FechaMovimiento"].notna()]
            ref = chunk["ID_ficticio"].map(reference_dates)
            chunk = chunk[chunk["FechaMovimiento"] <= ref]
            if chunk.empty:
                continue
        chunk["Movimiento"] = _clean_string_series(chunk["Movimiento"])
        chunk["Plan"] = _clean_string_series(chunk["Plan"])
        chunk["Canal"] = _clean_string_series(chunk["Canal"])

        id_col = "ID_ficticio"

        sums = chunk.groupby(id_col, observed=True)["CantidadMesesPagados"].agg(
            ["sum", "count"]
        )
        sums.columns = ["meses_pagados_sum", "movimientos_total"]
        numeric_sums = _accumulate_count_table(numeric_sums, sums)

        move_counts = (
            chunk.groupby([id_col, "Movimiento"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        plan_counts = (
            chunk.groupby([id_col, "Plan"], observed=True).size().unstack(fill_value=0)
        )
        canal_counts = (
            chunk.groupby([id_col, "Canal"], observed=True).size().unstack(fill_value=0)
        )
        count_tables["movimiento"] = _accumulate_count_table(
            count_tables["movimiento"], move_counts
        )
        count_tables["plan"] = _accumulate_count_table(
            count_tables["plan"], plan_counts
        )
        count_tables["canal"] = _accumulate_count_table(
            count_tables["canal"], canal_counts
        )

        move_min = chunk.groupby(id_col, observed=True)["FechaMovimiento"].min()
        move_max = chunk.groupby(id_col, observed=True)["FechaMovimiento"].max()
        cutoff_max = chunk.groupby(id_col, observed=True)["fechacorte_seguro"].max()
        due_max = chunk.groupby(id_col, observed=True)["FechaVencimiento"].max()

        min_move_date = _accumulate_min_series(min_move_date, move_min)
        max_move_date = _accumulate_max_series(max_move_date, move_max)
        max_cutoff_date = _accumulate_max_series(max_cutoff_date, cutoff_max)
        max_due_date = _accumulate_max_series(max_due_date, due_max)

    result = count_tables["movimiento"].copy()
    result.columns = [
        f"mov_{str(col).strip().lower().replace(' ', '_')}" for col in result.columns
    ]
    result["meses_pagados_sum"] = (
        numeric_sums["meses_pagados_sum"] if numeric_sums is not None else np.nan
    )
    result["movimientos_total"] = (
        numeric_sums["movimientos_total"] if numeric_sums is not None else np.nan
    )
    result["meses_pagados_mean"] = result["meses_pagados_sum"] / result[
        "movimientos_total"
    ].replace(0, np.nan)

    if count_tables["plan"] is not None:
        plan = count_tables["plan"].copy()
        plan.columns = [
            f"plan_{str(col).strip().lower().replace(' ', '_')}" for col in plan.columns
        ]
        result = result.join(plan, how="outer")

    if count_tables["canal"] is not None:
        canal = count_tables["canal"].copy()
        canal.columns = [
            f"canal_{str(col).strip().lower().replace(' ', '_')}"
            for col in canal.columns
        ]
        result = result.join(canal, how="outer")

    result = result.join(min_move_date.rename("fecha_primera_mov"), how="outer")
    result = result.join(max_move_date.rename("fecha_ultima_mov"), how="outer")
    result = result.join(max_cutoff_date.rename("fecha_corte_max"), how="outer")
    result = result.join(max_due_date.rename("fecha_vencimiento_max"), how="outer")

    result["n_planes"] = result.filter(like="plan_").gt(0).sum(axis=1)
    result["n_canales"] = result.filter(like="canal_").gt(0).sum(axis=1)
    result["share_abono"] = result.get("mov_abono", 0) / result[
        "movimientos_total"
    ].replace(0, np.nan)
    result["share_compra"] = result.get("mov_compra", 0) / result[
        "movimientos_total"
    ].replace(0, np.nan)
    result["share_cambio_plan"] = result.get("mov_cambio_de_plan", 0) / result[
        "movimientos_total"
    ].replace(0, np.nan)
    result["antiguedad_dias"] = (
        result["fecha_corte_max"] - result["fecha_primera_mov"]
    ).dt.days
    result["dias_desde_ultimo_mov"] = (
        result["fecha_corte_max"] - result["fecha_ultima_mov"]
    ).dt.days
    result["dias_hasta_vencimiento"] = (
        result["fecha_vencimiento_max"] - result["fecha_corte_max"]
    ).dt.days
    return result.reset_index()


def aggregate_transactions(
    path: Path,
    chunksize: int = CHUNK_SIZE,
    reference_dates: pd.Series | None = None,
) -> pd.DataFrame:
    usecols = [
        "ID_ficticio",
        "cartera",
        "fechacompra",
        "fechacorte_transaccion",
        "descategoria",
        "tipocompra",
        "precio_vta_perc",
    ]

    count_tables: dict[str, pd.DataFrame | None] = {
        "tipocompra": None,
        "descategoria": None,
    }
    numeric_sums: pd.DataFrame | None = None
    max_buy_date: pd.Series | None = None
    max_cutoff_date: pd.Series | None = None

    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        chunk = chunk.loc[:, ~chunk.columns.astype(str).str.match(r"^Unnamed")]
        chunk["precio_vta_perc"] = pd.to_numeric(
            chunk["precio_vta_perc"], errors="coerce"
        )
        for col in ["fechacompra", "fechacorte_transaccion"]:
            chunk[col] = pd.to_datetime(chunk[col], errors="coerce")
        if reference_dates is not None:
            event_date = chunk["fechacompra"].fillna(chunk["fechacorte_transaccion"])
            ref = chunk["ID_ficticio"].map(reference_dates)
            chunk = chunk[ref.notna() & event_date.notna()]
            ref = chunk["ID_ficticio"].map(reference_dates)
            event_date = chunk["fechacompra"].fillna(chunk["fechacorte_transaccion"])
            chunk = chunk[event_date <= ref]
            if chunk.empty:
                continue
        chunk["cartera"] = _clean_string_series(chunk["cartera"])
        chunk["descategoria"] = _clean_string_series(chunk["descategoria"])
        chunk["tipocompra"] = _clean_string_series(chunk["tipocompra"])

        id_col = "ID_ficticio"
        sums = chunk.groupby(id_col, observed=True)["precio_vta_perc"].agg(
            ["sum", "mean", "count"]
        )
        sums.columns = ["precio_vta_sum", "precio_vta_mean", "transacciones_total"]
        numeric_sums = _accumulate_count_table(numeric_sums, sums)

        tipo_counts = (
            chunk.groupby([id_col, "tipocompra"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        desc_counts = (
            chunk.groupby([id_col, "descategoria"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        count_tables["tipocompra"] = _accumulate_count_table(
            count_tables["tipocompra"], tipo_counts
        )
        count_tables["descategoria"] = _accumulate_count_table(
            count_tables["descategoria"], desc_counts
        )

        buy_max = chunk.groupby(id_col, observed=True)["fechacompra"].max()
        cutoff_max = chunk.groupby(id_col, observed=True)[
            "fechacorte_transaccion"
        ].max()
        max_buy_date = _accumulate_max_series(max_buy_date, buy_max)
        max_cutoff_date = _accumulate_max_series(max_cutoff_date, cutoff_max)

    result = numeric_sums.copy() if numeric_sums is not None else pd.DataFrame()
    if count_tables["tipocompra"] is not None:
        tipo = count_tables["tipocompra"].copy()
        tipo.columns = [f"trx_{str(col).strip().lower()}" for col in tipo.columns]
        result = result.join(tipo, how="outer")
    if count_tables["descategoria"] is not None:
        desc = count_tables["descategoria"].copy()
        desc.columns = [
            f"cat_{str(col).strip().lower().replace(' ', '_').replace(',', '')}"
            for col in desc.columns
        ]
        result = result.join(desc, how="outer")

    result = result.join(max_buy_date.rename("fecha_ultima_compra"), how="outer")
    result = result.join(
        max_cutoff_date.rename("fecha_corte_transaccion_max"), how="outer"
    )

    result["share_credito"] = result.get("trx_credito", 0) / result[
        "transacciones_total"
    ].replace(0, np.nan)
    result["share_contado"] = result.get("trx_contado", 0) / result[
        "transacciones_total"
    ].replace(0, np.nan)
    result["dias_desde_ultima_compra"] = (
        result["fecha_corte_transaccion_max"] - result["fecha_ultima_compra"]
    ).dt.days
    result["n_categorias_desc"] = result.filter(like="cat_").gt(0).sum(axis=1)
    return result.reset_index()


def load_sociodemographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    if "" in df.columns:
        df = df.drop(columns=[""])
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = _clean_string_series(df[col])
    return df.drop_duplicates(subset=["ID_ficticio"], keep="first")


def _normalize_categorical_na(x: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    if not categorical:
        return x
    x = x.copy()
    for col in categorical:
        if col in x.columns:
            x[col] = x[col].astype("object")
            x[col] = x[col].where(pd.notna(x[col]), np.nan)
    return x


def create_churn_target_from_movements(
    customer_path: Path,
    horizon_months: int = DEFAULT_CHURN_HORIZON_MONTHS,
    chunksize: int = CHUNK_SIZE,
) -> pd.DataFrame:
    usecols = ["ID_ficticio", "FechaMovimiento", "Movimiento"]
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        customer_path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        chunk = chunk.loc[:, ~chunk.columns.astype(str).str.match(r"^Unnamed")]
        chunk["FechaMovimiento"] = pd.to_datetime(
            chunk["FechaMovimiento"], errors="coerce"
        )
        chunk["Movimiento"] = _clean_string_series(chunk["Movimiento"]).str.lower()
        chunk = chunk[chunk["FechaMovimiento"].notna()]
        chunk = chunk[chunk["Movimiento"].isin(["abono", "compra", "cambio de plan"])]
        if not chunk.empty:
            chunks.append(chunk[["ID_ficticio", "FechaMovimiento", "Movimiento"]])

    if not chunks:
        return pd.DataFrame(columns=["ID_ficticio", "desercion"])

    events = pd.concat(chunks, ignore_index=True)
    global_max = events["FechaMovimiento"].max()
    observation_end = global_max - pd.DateOffset(months=horizon_months)

    observed = events[events["FechaMovimiento"] <= observation_end].copy()
    if observed.empty:
        return pd.DataFrame(columns=["ID_ficticio", "desercion"])

    reference = observed.groupby("ID_ficticio", as_index=False)["FechaMovimiento"].max()
    reference = reference.rename(columns={"FechaMovimiento": "fecha_referencia_modelo"})
    reference["fecha_limite_ventana"] = reference[
        "fecha_referencia_modelo"
    ] + pd.DateOffset(months=horizon_months)

    events_with_ref = events.merge(
        reference[["ID_ficticio", "fecha_referencia_modelo", "fecha_limite_ventana"]],
        on="ID_ficticio",
        how="inner",
    )
    in_future_window = (
        events_with_ref["FechaMovimiento"] > events_with_ref["fecha_referencia_modelo"]
    ) & (events_with_ref["FechaMovimiento"] <= events_with_ref["fecha_limite_ventana"])
    future_counts = (
        events_with_ref.loc[in_future_window]
        .groupby("ID_ficticio")
        .size()
        .rename("movimientos_siguientes_ventana")
    )

    target = reference.copy()
    target = target.merge(future_counts, on="ID_ficticio", how="left")
    target["movimientos_siguientes_ventana"] = (
        target["movimientos_siguientes_ventana"].fillna(0).astype(int)
    )
    target["tiene_movimientos_siguientes"] = (
        target["movimientos_siguientes_ventana"] > 0
    ).astype(int)
    target["desercion"] = (target["movimientos_siguientes_ventana"] == 0).astype(int)
    target["horizonte_desercion_meses"] = horizon_months
    return target


def create_churn_target_from_renewal(
    customer_path: Path,
    grace_after_days: int = DEFAULT_RENEWAL_GRACE_AFTER_DAYS,
    chunksize: int = CHUNK_SIZE,
) -> pd.DataFrame:
    usecols = ["ID_ficticio", "FechaMovimiento", "FechaVencimiento", "Movimiento"]
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        customer_path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        chunk = chunk.loc[:, ~chunk.columns.astype(str).str.match(r"^Unnamed")]
        chunk["FechaMovimiento"] = pd.to_datetime(
            chunk["FechaMovimiento"], errors="coerce"
        )
        chunk["FechaVencimiento"] = pd.to_datetime(
            chunk["FechaVencimiento"], errors="coerce"
        )
        chunk["Movimiento"] = _clean_string_series(chunk["Movimiento"]).str.lower()
        chunk = chunk[chunk["FechaMovimiento"].notna()]
        chunk = chunk[chunk["Movimiento"].isin(["abono", "compra", "cambio de plan"])]
        if not chunk.empty:
            chunks.append(
                chunk[
                    [
                        "ID_ficticio",
                        "FechaMovimiento",
                        "FechaVencimiento",
                        "Movimiento",
                    ]
                ]
            )

    if not chunks:
        return pd.DataFrame(columns=["ID_ficticio", "desercion"])

    events = pd.concat(chunks, ignore_index=True)
    max_event_date = events["FechaMovimiento"].max()
    due_observation_limit = max_event_date - pd.Timedelta(days=grace_after_days)

    candidate_due = events[
        events["FechaVencimiento"].notna()
        & (events["FechaVencimiento"] <= due_observation_limit)
    ].copy()
    if candidate_due.empty:
        return pd.DataFrame(columns=["ID_ficticio", "desercion"])

    due_per_client = (
        candidate_due.groupby("ID_ficticio", as_index=False)["FechaVencimiento"]
        .max()
        .rename(columns={"FechaVencimiento": "fecha_vencimiento_objetivo"})
    )
    history_for_reference = events.merge(due_per_client, on="ID_ficticio", how="inner")
    history_for_reference = history_for_reference[
        history_for_reference["FechaMovimiento"]
        <= history_for_reference["fecha_vencimiento_objetivo"]
    ]
    if history_for_reference.empty:
        return pd.DataFrame(columns=["ID_ficticio", "desercion"])

    reference_dates = (
        history_for_reference.groupby("ID_ficticio", as_index=False)["FechaMovimiento"]
        .max()
        .rename(columns={"FechaMovimiento": "fecha_referencia_modelo"})
    )
    reference = due_per_client.merge(reference_dates, on="ID_ficticio", how="inner")

    reference["fecha_inicio_ventana_renovacion"] = reference[
        "fecha_vencimiento_objetivo"
    ] + pd.Timedelta(days=1)
    reference["fecha_fin_ventana_renovacion"] = reference[
        "fecha_vencimiento_objetivo"
    ] + pd.Timedelta(days=grace_after_days)

    events_with_ref = events.merge(
        reference[
            [
                "ID_ficticio",
                "fecha_referencia_modelo",
                "fecha_vencimiento_objetivo",
                "fecha_inicio_ventana_renovacion",
                "fecha_fin_ventana_renovacion",
            ]
        ],
        on="ID_ficticio",
        how="inner",
    )
    in_renewal_window = (
        (
            events_with_ref["FechaMovimiento"]
            > events_with_ref["fecha_referencia_modelo"]
        )
        & (
            events_with_ref["FechaMovimiento"]
            > events_with_ref["fecha_vencimiento_objetivo"]
        )
        & (
            events_with_ref["FechaMovimiento"]
            <= events_with_ref["fecha_fin_ventana_renovacion"]
        )
    )
    renewal_counts = (
        events_with_ref.loc[in_renewal_window]
        .groupby("ID_ficticio")
        .size()
        .rename("movimientos_ventana_renovacion")
    )

    target = reference.merge(renewal_counts, on="ID_ficticio", how="left")
    target["movimientos_ventana_renovacion"] = (
        target["movimientos_ventana_renovacion"].fillna(0).astype(int)
    )
    target["renovo_en_ventana"] = (target["movimientos_ventana_renovacion"] > 0).astype(
        int
    )
    target["desercion"] = (target["renovo_en_ventana"] == 0).astype(int)
    target["definicion_target"] = "no_renovacion_seguro"
    target["horizonte_desercion_meses"] = grace_after_days / 30.0
    return target


def build_client_dataset(
    churn_horizon_months: int = DEFAULT_CHURN_HORIZON_MONTHS,
    target_definition: str = "renewal",
    renewal_grace_after_days: int = DEFAULT_RENEWAL_GRACE_AFTER_DAYS,
) -> pd.DataFrame:
    if target_definition == "renewal":
        target = create_churn_target_from_renewal(
            CUSTOMER_FILE,
            grace_after_days=renewal_grace_after_days,
        )
    elif target_definition == "movements":
        target = create_churn_target_from_movements(
            CUSTOMER_FILE, horizon_months=churn_horizon_months
        )
    else:
        raise ValueError("target_definition debe ser 'renewal' o 'movements'.")

    if target.empty:
        raise ValueError(
            "No fue posible construir la variable desercion con los datos de movimientos."
        )

    reference_dates = target.set_index("ID_ficticio")["fecha_referencia_modelo"]

    socio = load_sociodemographics(SOCIO_FILE)
    clientes = aggregate_client_activity(CUSTOMER_FILE, reference_dates=reference_dates)
    trans = aggregate_transactions(TRANS_FILE, reference_dates=reference_dates)

    merged = socio.merge(target, on="ID_ficticio", how="inner")
    merged = merged.merge(clientes, on="ID_ficticio", how="left", suffixes=("", "_mov"))
    merged = merged.merge(trans, on="ID_ficticio", how="left", suffixes=("", "_trx"))

    string_cols = [
        "escolaridad",
        "estado",
        "estadocivil",
        "genero",
        "puntualidad",
        "rangoedad",
        "rangoingreso",
        "rangolineacredito",
    ]
    for col in string_cols:
        if col in merged.columns:
            merged[col] = _clean_string_series(merged[col])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "merged_client_dataset.csv", index=False)

    return merged


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = (
        df.loc[df["desercion"].notna(), "desercion"]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    data.plot(kind="bar", ax=ax, color=["#3B82F6", "#EF4444"])
    ax.set_title("Distribucion de desercion")
    ax.set_xlabel("Desercion")
    ax.set_ylabel("Clientes")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=150)
    plt.close(fig)


def sample_for_modeling(
    df: pd.DataFrame, target_col: str, max_rows: int = MAX_MODEL_ROWS
) -> pd.DataFrame:
    model_df = df.loc[df[target_col].notna()].copy()
    if len(model_df) <= max_rows:
        return model_df
    sampled, _ = train_test_split(
        model_df,
        train_size=max_rows,
        stratify=model_df[target_col],
        random_state=RANDOM_STATE,
    )
    return sampled.copy()


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {
        "ID_ficticio",
        "desercion",
        "definicion_target",
        "fecha_referencia_modelo",
        "fecha_limite_ventana",
        "fecha_vencimiento_objetivo",
        "fecha_inicio_ventana_renovacion",
        "fecha_fin_ventana_renovacion",
        "movimientos_siguientes_ventana",
        "movimientos_ventana_renovacion",
        "horizonte_desercion_meses",
        "tiene_movimientos_siguientes",
        "renovo_en_ventana",
    }
    categorical = [
        col
        for col in [
            "escolaridad",
            "estado",
            "estadocivil",
            "puntualidad",
            "rangoedad",
            "rangoingreso",
            "rangolineacredito",
        ]
        if col in df.columns
    ]
    numeric = [
        col
        for col in df.columns
        if col not in exclude
        and col not in categorical
        and pd.api.types.is_numeric_dtype(df[col])
        and col != "desercion"
    ]
    return numeric, categorical


def evaluate_classifier(model, x_test, y_test, name: str) -> dict[str, float]:
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
    }
    print("=" * 100)
    print(name)
    print(metrics)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4))
    return metrics


def _build_logistic_preprocessor(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_logistic_split(
    df: pd.DataFrame,
    split_mode: str = "random",
    drop_features: set[str] | None = None,
    max_rows: int = 120_000,
) -> dict[str, float]:
    numeric, categorical = build_feature_lists(df)
    if drop_features:
        numeric = [c for c in numeric if c not in drop_features]
        categorical = [c for c in categorical if c not in drop_features]
    features = numeric + categorical
    if not features:
        raise ValueError("No hay features para evaluar el split.")

    work = df.loc[df["desercion"].notna(), features + ["desercion"]].copy()
    work = sample_for_modeling(work, "desercion", max_rows=max_rows)
    x = _normalize_categorical_na(work[features], categorical)
    y = work["desercion"].astype(int)

    if split_mode == "temporal" and "fecha_referencia_modelo" in df.columns:
        dates = df.loc[work.index, "fecha_referencia_modelo"]
        order = np.argsort(dates.to_numpy(dtype="datetime64[ns]"))
        n_train = int(0.8 * len(work))
        train_idx = work.index[order[:n_train]]
        test_idx = work.index[order[n_train:]]
        x_train, x_test = x.loc[train_idx], x.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y,
            )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    model = Pipeline(
        steps=[
            ("prep", _build_logistic_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "split": split_mode,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
    }


def run_leakage_diagnostics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = df.loc[df["desercion"].notna(), "desercion"].astype(int)
    numeric, _ = build_feature_lists(df)

    leak_keyword_cols = [
        col
        for col in df.columns
        if any(
            token in col
            for token in [
                "siguientes",
                "ventana",
                "renovo",
                "vencimiento_objetivo",
                "limite_ventana",
            ]
        )
    ]

    univariate_rows: list[dict[str, float | str]] = []
    for col in numeric:
        s = df.loc[y.index, col]
        if s.notna().sum() < 500:
            continue
        filled = s.fillna(s.median())
        try:
            auc = float(roc_auc_score(y, filled))
        except ValueError:
            continue
        strength = max(auc, 1 - auc)
        univariate_rows.append(
            {
                "feature": col,
                "auc": auc,
                "auc_strength": strength,
            }
        )

    if univariate_rows:
        univariate_df = pd.DataFrame(univariate_rows).sort_values(
            "auc_strength", ascending=False
        )
    else:
        univariate_df = pd.DataFrame(columns=["feature", "auc", "auc_strength"])

    suspicious_uni = (
        univariate_df.loc[univariate_df["auc_strength"] >= 0.95, "feature"].tolist()
        if not univariate_df.empty
        else []
    )

    split_random = evaluate_logistic_split(df, split_mode="random")
    split_temporal = evaluate_logistic_split(df, split_mode="temporal")
    split_no_suspects = evaluate_logistic_split(
        df,
        split_mode="temporal",
        drop_features=set(suspicious_uni),
    )

    summary_rows = [
        {
            "check": "potential_leak_columns_present",
            "value": len(leak_keyword_cols),
            "detail": ", ".join(leak_keyword_cols[:15]),
        },
        {
            "check": "suspicious_univariate_features_auc>=0.95",
            "value": len(suspicious_uni),
            "detail": ", ".join(suspicious_uni[:15]),
        },
        {
            "check": "random_split_auc",
            "value": split_random["roc_auc"],
            "detail": "",
        },
        {
            "check": "temporal_split_auc",
            "value": split_temporal["roc_auc"],
            "detail": "",
        },
        {
            "check": "temporal_split_auc_without_suspicious",
            "value": split_no_suspects["roc_auc"],
            "detail": "",
        },
    ]
    summary = pd.DataFrame(summary_rows)

    print("=" * 100)
    print("Leakage diagnostics")
    print(summary.to_string(index=False))
    if not univariate_df.empty:
        print("Top 10 univariate AUC strength features:")
        print(univariate_df.head(10).to_string(index=False))

    summary.to_csv(output_dir / "leakage_diagnostics_summary.csv", index=False)
    if not univariate_df.empty:
        univariate_df.to_csv(output_dir / "leakage_univariate_auc.csv", index=False)

    return summary


def fit_logistic_model(df: pd.DataFrame) -> dict[str, float]:
    numeric, categorical = build_feature_lists(df)
    features = numeric + categorical
    if not features:
        raise ValueError(
            "No hay variables disponibles para entrenar la regresion logistica."
        )

    work = df.loc[df["desercion"].notna(), features + ["desercion"]].copy()
    if work.empty:
        raise ValueError(
            "No hay observaciones con desercion disponible para entrenar el modelo."
        )

    work = sample_for_modeling(work, "desercion", max_rows=MAX_MODEL_ROWS)

    x = _normalize_categorical_na(work[features], categorical)
    y = work["desercion"].astype(int)
    if y.nunique() < 2:
        raise ValueError(
            "La variable objetivo desercion tiene una sola clase en la muestra."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = _build_logistic_preprocessor(numeric, categorical)

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    metrics = evaluate_classifier(
        model, x_test, y_test, "LogisticRegression (baseline interpretable)"
    )
    return metrics


def fit_hist_gradient_boosting_model(df: pd.DataFrame) -> dict[str, float]:
    numeric, categorical = build_feature_lists(df)
    features = numeric + categorical
    work = df.loc[df["desercion"].notna(), features + ["desercion"]].copy()
    work = sample_for_modeling(work, "desercion", max_rows=MAX_MODEL_ROWS)

    x = _normalize_categorical_na(work[features], categorical)
    y = work["desercion"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.05,
                    max_iter=250,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    metrics = evaluate_classifier(
        model, x_test, y_test, "HistGradientBoosting (nonlinear alternative)"
    )
    return metrics


def latent_variable_analysis(df: pd.DataFrame) -> pd.DataFrame:
    latent_candidates = [
        "movimientos_total",
        "meses_pagados_sum",
        "meses_pagados_mean",
        "share_abono",
        "share_compra",
        "share_cambio_plan",
        "antiguedad_dias",
        "dias_desde_ultimo_mov",
        "dias_hasta_vencimiento",
        "transacciones_total",
        "precio_vta_mean",
        "share_credito",
        "share_contado",
        "dias_desde_ultima_compra",
        "n_canales",
        "n_planes",
        "n_categorias_desc",
    ]
    latent_cols = [col for col in latent_candidates if col in df.columns]
    latent_df = df.loc[df["desercion"].notna(), latent_cols + ["desercion"]].copy()
    latent_df = latent_df.dropna(subset=latent_cols, how="all")

    if len(latent_cols) < 3 or latent_df.empty:
        print(
            "No hay suficientes indicadores numericos para un factor analysis estable."
        )
        return pd.DataFrame()

    x = latent_df[latent_cols].fillna(latent_df[latent_cols].median(numeric_only=True))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    n_components = 2 if len(latent_cols) >= 4 else 1
    fa = FactorAnalysis(n_components=n_components, random_state=RANDOM_STATE)
    scores = fa.fit_transform(x_scaled)

    loadings = pd.DataFrame(
        fa.components_.T,
        index=latent_cols,
        columns=[f"factor_{i + 1}" for i in range(n_components)],
    )
    print("=" * 100)
    print("Factor analysis loadings")
    print(loadings.sort_values(by=loadings.columns[0], ascending=False).to_string())

    latent_score = scores[:, 0]
    target = latent_df["desercion"].astype(int).to_numpy()
    if np.corrcoef(latent_score, target)[0, 1] < 0:
        latent_score = -latent_score

    latent_df = latent_df.copy()
    latent_df["riesgo_latente"] = latent_score
    latent_auc = roc_auc_score(target, latent_df["riesgo_latente"])
    print(f"AUC using only latent risk score: {latent_auc:.4f}")
    return loadings


def save_feature_snapshot(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        col
        for col in [
            "ID_ficticio",
            "desercion",
            "fecha_referencia_modelo",
            "fecha_limite_ventana",
            "movimientos_siguientes_ventana",
            "escolaridad",
            "estado",
            "estadocivil",
            "genero",
            "puntualidad",
            "rangoedad",
            "rangoingreso",
            "rangolineacredito",
            "movimientos_total",
            "meses_pagados_sum",
            "dias_desde_ultimo_mov",
            "transacciones_total",
            "share_credito",
            "dias_desde_ultima_compra",
        ]
        if col in df.columns
    ]
    snapshot = df.loc[df["desercion"].notna(), cols].head(2000)
    snapshot.to_csv(output_dir / "snapshot_modelado.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    socio_sample = read_sample(SOCIO_FILE)
    clientes_sample = read_sample(CUSTOMER_FILE)
    trans_sample = read_sample(TRANS_FILE)

    print_eda("Sociodemograficos (muestra)", socio_sample)
    print_eda("Clientes / movimientos (muestra)", clientes_sample)
    print_eda("Transacciones (muestra)", trans_sample)

    merged = build_client_dataset(
        churn_horizon_months=DEFAULT_CHURN_HORIZON_MONTHS,
        target_definition="renewal",
        renewal_grace_after_days=DEFAULT_RENEWAL_GRACE_AFTER_DAYS,
    )
    print("=" * 100)
    print("Base final unificada")
    print("shape:", merged.shape)
    print("desercion distribution:")
    print(merged["desercion"].value_counts(dropna=False).to_string())
    print("missing ratio in key engineered features:")
    key_cols = [
        col
        for col in [
            "movimientos_total",
            "meses_pagados_sum",
            "dias_desde_ultimo_mov",
            "transacciones_total",
            "share_credito",
            "dias_desde_ultima_compra",
        ]
        if col in merged.columns
    ]
    if key_cols:
        print(merged[key_cols].isna().mean().sort_values(ascending=False).to_string())

    plot_target_distribution(merged, OUTPUT_DIR)
    save_feature_snapshot(merged, OUTPUT_DIR)
    run_leakage_diagnostics(merged, OUTPUT_DIR)

    logistic_metrics = fit_logistic_model(merged)
    gbdt_metrics = fit_hist_gradient_boosting_model(merged)
    latent_loadings = latent_variable_analysis(merged)

    summary = pd.DataFrame([logistic_metrics, gbdt_metrics])
    summary.to_csv(OUTPUT_DIR / "metricas_modelos.csv", index=False)
    if not latent_loadings.empty:
        latent_loadings.to_csv(OUTPUT_DIR / "latent_loadings.csv")

    print("=" * 100)
    print("Resumen de metricas")
    print(summary.to_string(index=False))
    print(f"Outputs guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
