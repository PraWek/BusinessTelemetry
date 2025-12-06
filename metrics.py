"""
metrics.py

Воронки (funnel), KPI, Sankey-переходы и дневные конверсии.
"""
from typing import List, Dict, Optional
import pandas as pd


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """
    Безопасное деление: возвращает None если знаменатель <= 0 или невозможно преобразовать.
    """
    try:
        d = float(denominator)
    except Exception:
        return None
    if d <= 0:
        return None
    return float(numerator) / d


def compute_funnel(
    df: pd.DataFrame,
    steps: List[str],
    session_col: str = "sessionid",
    userid_col: str = "userid",
    ts_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Для каждого шага возвращает:
      - sessions_reached: число уникальных sessionid, в которых шаг был пройден в порядке
      - unique_users_reached: число уникальных userid среди этих сессий

    Логика:
      для каждой сессии мы проверяем, встречаются ли шаги в заданном порядке (не обязательно подряд),
      и если да — считаем сессию как достигшую соответствующих шагов.
    """
    if session_col not in df.columns:
        raise ValueError(f"{session_col} not found in df")
    df_local = df.copy()
    if ts_col in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local[ts_col]):
        df_local[ts_col] = pd.to_datetime(df_local[ts_col], errors="coerce")
    df_local = df_local.sort_values([session_col, ts_col])

    rows = []
    # Для каждой сессии проверяем последовательность
    for sid, sgrp in df_local.groupby(session_col):
        actions = list(sgrp["action"])
        user = sgrp[userid_col].iloc[0] if userid_col in sgrp.columns else None
        pos = 0
        reached = set()
        for step in steps:
            try:
                i = actions.index(step, pos)
                reached.add(step)
                pos = i + 1
            except ValueError:
                break
        for step in reached:
            rows.append({"sessionid": sid, "userid": user, "step": step})

    if not rows:
        return pd.DataFrame({"step": [], "sessions_reached": [], "unique_users_reached": []})

    reached_df = pd.DataFrame(rows)
    agg = (
        reached_df.groupby("step")
        .agg(sessions_reached=("sessionid", "nunique"), unique_users_reached=("userid", "nunique"))
        .reset_index()
    )
    # сохранить порядок шагов
    agg["step_number"] = agg["step"].apply(lambda s: steps.index(s) if s in steps else -1)
    agg = agg.sort_values("step_number").drop(columns="step_number")
    return agg


def compute_kpis_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    orders_action: str = "checkout",
    confirmation_col: Optional[str] = "confirmation"
) -> pd.DataFrame:
    """
    Возвращает DataFrame с KPI по дням:
      - gmv (sum checkout_value)
      - orders_count (по action == orders_action или confirmation_col)
      - buyers_count (unique userid)
      - sessions_count (unique sessionid)
      - aov (gmv / orders_count, безопасно)
      - dau
    Ожидается, что df[date_col] — либо python date, либо datetime (будет преобразовано в date).
    """
    df_local = df.copy()
    # ensure date_col is date
    if date_col in df_local.columns and pd.api.types.is_datetime64_any_dtype(df_local[date_col]):
        df_local[date_col] = df_local[date_col].dt.date
    elif date_col not in df_local.columns and "timestamp" in df_local.columns:
        df_local[date_col] = df_local["timestamp"].dt.date

    # orders
    if confirmation_col and confirmation_col in df_local.columns:
        orders_series = df_local[confirmation_col].astype(bool)
        orders = df_local.loc[orders_series].groupby(date_col).size().rename("orders_count").reset_index()
    else:
        orders = (
            df_local.groupby(date_col)
            .apply(lambda g: (g["action"] == orders_action).sum())
            .rename("orders_count")
            .reset_index()
        )

    gmv = df_local.groupby(date_col)["checkout_value"].sum().reset_index(name="gmv")
    sessions = df_local.groupby(date_col)["sessionid"].nunique().reset_index(name="sessions_count")
    buyers = df_local.groupby(date_col)["userid"].nunique().reset_index(name="buyers_count")
    dau = df_local.groupby(date_col)["userid"].nunique().reset_index(name="dau")

    kpis = orders.merge(gmv, on=date_col, how="outer").merge(sessions, on=date_col, how="outer").merge(buyers, on=date_col, how="outer")
    kpis = kpis.merge(dau, on=date_col, how="left")
    kpis = kpis.fillna({ "orders_count": 0, "gmv": 0, "sessions_count": 0, "buyers_count": 0, "dau": 0 })

    # safe AOV
    kpis["aov"] = kpis.apply(lambda r: safe_divide(r["gmv"], r["orders_count"]), axis=1)
    return kpis


def compute_sankey_transitions(
    df: pd.DataFrame,
    step_map: Dict[str, int],
    session_col: str = "sessionid",
    ts_col: str = "timestamp",
    require_step_increase: bool = True
) -> pd.DataFrame:
    """
    Считает переходы (action -> next_action) внутри сессий (только соседние события).
    Возвращает DataFrame columns=['action','next_action','users'] — число уникальных пользователей,
    совершивших такой переход внутри одной сессии.

    Если require_step_increase=True, то учитываются только переходы, где mapped(next) >= mapped(current),
    чтобы поддержать идею "продвижения по воронке". Можно выставить False, если нужны все переходы.
    """
    if session_col not in df.columns:
        raise ValueError(f"{session_col} not found in df")
    df_local = df.copy()
    if ts_col in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local[ts_col]):
        df_local[ts_col] = pd.to_datetime(df_local[ts_col], errors="coerce")
    df_local = df_local.sort_values([session_col, ts_col])
    df_local["next_action"] = df_local.groupby(session_col)["action"].shift(-1)
    df_local["current_step"] = df_local["action"].map(step_map)
    df_local["next_step"] = df_local["next_action"].map(step_map)
    # drop where mapping failed
    df_local = df_local[df_local["next_step"].notna() & df_local["current_step"].notna()]
    if require_step_increase:
        df_local = df_local[df_local["next_step"] >= df_local["current_step"]]
    sankey = df_local.groupby(["action", "next_action"])["userid"].nunique().reset_index(name="users")
    sankey = sankey.sort_values(["action", "next_action"]).reset_index(drop=True)
    return sankey


def compute_conversion_daily(df: pd.DataFrame, steps: List[str], date_col: str = "date") -> pd.DataFrame:
    """
    Для списка steps (в порядке) вычисляет дневные количества уникальных пользователей на каждом шаге
    и конверсии между соседними шагами, а также полную конверсию (последний/первый).
    Возвращает DataFrame с date и CR колонками. Если denominator == 0 -> NaN.
    """
    df_local = df.copy()
    if date_col in df_local.columns and pd.api.types.is_datetime64_any_dtype(df_local[date_col]):
        df_local[date_col] = df_local[date_col].dt.date
    elif date_col not in df_local.columns and "timestamp" in df_local.columns:
        df_local[date_col] = df_local["timestamp"].dt.date

    df_f = df_local[df_local["action"].isin(steps)].copy()
    daily_steps = df_f.groupby([date_col, "action"])["userid"].nunique().reset_index()
    pivot = daily_steps.pivot(index=date_col, columns="action", values="userid").fillna(0)
    # ensure all steps exist
    for s in steps:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[steps]  # re-order columns to match steps

    out = pivot.reset_index()
    # compute ratios
    for i in range(len(steps) - 1):
        a = steps[i]
        b = steps[i + 1]
        out[f"cr_{a}_to_{b}"] = out.apply(lambda r: safe_divide(r[b], r[a]), axis=1)
    out["cr_full"] = out.apply(lambda r: safe_divide(r[steps[-1]], r[steps[0]]), axis=1)
    return out
