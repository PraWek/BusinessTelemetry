"""
feature_engineering.py

Создание признаков: сдвиги внутри сессий, product->cart transitions,
basket size, average times и другие session-aware признаки.
"""
from typing import List, Optional
import pandas as pd


def add_session_shifts(df: pd.DataFrame, session_col: str = "sessionid", ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Добавляет prev/next action и prev/next timestamp ВНУТРИ каждой сессии.
    Гарантирует сортировку по session_col и ts_col перед shift.
    """
    if session_col not in df.columns:
        raise ValueError(f"{session_col} not found in df")
    df = df.copy()
    if ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values([session_col, ts_col])
    df["prev_action_in_session"] = df.groupby(session_col)["action"].shift(1)
    df["next_action_in_session"] = df.groupby(session_col)["action"].shift(-1)
    df["prev_ts_in_session"] = df.groupby(session_col)[ts_col].shift(1)
    df["next_ts_in_session"] = df.groupby(session_col)[ts_col].shift(-1)
    return df


def compute_session_step_number(df: pd.DataFrame, session_col: str = "sessionid", ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Вычисляет порядковый номер события внутри сессии (1,2,3,...).
    """
    df = df.copy()
    if ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values([session_col, ts_col])
    df["session_step_number"] = df.groupby(session_col).cumcount() + 1
    return df


def product_to_cart_transitions(df: pd.DataFrame, session_col: str = "sessionid", ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Возвращает DataFrame переходов product -> cart внутри сессии.
    Каждая строка — событие cart, у которого предыдущим action в той же сессии был product.
    """
    df = add_session_shifts(df, session_col=session_col, ts_col=ts_col)
    mask = (df["action"] == "cart") & (df["prev_action_in_session"] == "product")
    res = df.loc[mask].copy()
    return res[["userid", session_col, "prev_action_in_session", "action", "timestamp"]]


def compute_basket_size_and_avg_time(df: pd.DataFrame, session_col: str = "sessionid", ts_col: str = "timestamp") -> pd.DataFrame:
    """
    - product_to_cart внутри сессии (1/0)
    - basket_size: количество product->cart событий в сессии
    - avg_time_between_cart_and_checkout: в секундах для сессии (см. ниже)
    """
    df = df.copy()
    if ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values([session_col, ts_col])

    # product -> cart inside session
    df["product_to_cart"] = ((df["action"] == "cart") & (df.groupby(session_col)["action"].shift(1) == "product")).astype(int)
    # basket size per session
    df["basket_size"] = df.groupby(session_col)["product_to_cart"].transform("sum")

    # average time between cart and next checkout inside session
    df_cart_checkout = df[df["action"].isin(["cart", "checkout"])].copy()
    df_cart_checkout = df_cart_checkout.sort_values([session_col, ts_col])
    # diff gives time since previous cart/checkout in same session
    df_cart_checkout["time_since_prev"] = df_cart_checkout.groupby(session_col)[ts_col].diff().dt.total_seconds()
    # For checkout rows, time_since_prev represents time from previous cart/checkout to checkout.
    avg_time = (
        df_cart_checkout[df_cart_checkout["action"] == "checkout"]
        .groupby(session_col)["time_since_prev"]
        .mean()
    )
    df["avg_time_between_cart_and_checkout"] = df[session_col].map(avg_time).fillna(0)
    return df


def first_action_per_session(df: pd.DataFrame, session_col: str = "sessionid", ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Возвращает DataFrame с первым action и временем в каждой сессии.
    """
    if session_col not in df.columns:
        raise ValueError(f"{session_col} not found")
    df_local = df.copy()
    if ts_col in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local[ts_col]):
        df_local[ts_col] = pd.to_datetime(df_local[ts_col], errors="coerce")
    firsts = df_local.sort_values([session_col, ts_col]).groupby(session_col).first().reset_index()
    return firsts
