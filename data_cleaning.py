"""
data_cleaning.py

Чтение и базовая очистка данных, работа с timestamp и сессиями.

Функции возвращают чистые DataFrame и/или сессионные агрегаты.
"""
from typing import Tuple, Optional
import pandas as pd


def read_telemetry_csv(path: str, ts_col: str = "timestamp", drop_unnamed: bool = True) -> pd.DataFrame:
    """
    Прочитать CSV и привести timestamp к datetime.
    Удаляет 'Unnamed: 0' если есть.
    """
    df = pd.read_csv(path)
    if drop_unnamed and "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # приведение timestamp
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["date"] = df[ts_col].dt.date
    else:
        df["date"] = pd.NaT
    return df


def parse_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Гарантирует тип datetime для timestamp и создает колонку date (python date).
    Возвращаем копию.
    """
    df = df.copy()
    if ts_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["date"] = df[ts_col].dt.date
    else:
        df["date"] = pd.NaT
    return df


def compute_session_aggregates(
    df: pd.DataFrame,
    session_col: str = "sessionid",
    ts_col: str = "timestamp",
    keep_events: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Вычисляет таблицу сессий:
      session_start, session_end, session_duration (в секундах), events_count
    И возвращает:
      - events_df (если keep_events=True) — оригинальный df с колонкой session_duration (подмножество колонок сессий замержено),
      - sessions_df — DataFrame со сессионными агрегатами.

    ВАЖНО: сессии вычисляются по session_col, поэтому session_col должен присутствовать.
    """
    if session_col not in df.columns:
        raise ValueError(f"Column {session_col} not found in df")

    df_local = df.copy()
    # гарантируем datetime
    if ts_col in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local[ts_col]):
        df_local[ts_col] = pd.to_datetime(df_local[ts_col], errors="coerce")

    sessions = (
        df_local.groupby(session_col, as_index=False)
        .agg(
            session_start=(ts_col, "min"),
            session_end=(ts_col, "max"),
            events_count=(ts_col, "count")
        )
    )
    sessions["session_duration"] = (sessions["session_end"] - sessions["session_start"]).dt.total_seconds().fillna(0)
    # merge session_duration обратно в события (выравнивание по sessionid)
    if keep_events:
        df_out = df_local.merge(sessions[[session_col, "session_duration"]], on=session_col, how="left")
        return df_out, sessions
    else:
        return df_local, sessions


def safe_fill_category(df: pd.DataFrame, col: str = "category", fill_value: str = "unknown") -> pd.DataFrame:
    """
    Заменяет NaN в колонке category на 'unknown' (по умолчанию).
    """
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].fillna(fill_value)
    return df


def safe_assign_column(target_df: pd.DataFrame, source_series: pd.Series, col_name: str) -> pd.DataFrame:
    """
    Безопасно присваивает колонку source_series к target_df, выравнивая по индексам.
    Полезно чтобы избежать некорректного переписывания session_duration.
    """
    out = target_df.copy()
    out[col_name] = source_series.reindex(out.index)
    return out
