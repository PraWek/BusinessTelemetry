"""
cohorts.py

Функции для когортного анализа и разметки активности пользователей (New/Returning/Churn-risk/Active).
"""
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd


def label_user_activity_segment(
    df: pd.DataFrame,
    first_visit_col: str = "user_first_visit_date",
    ts_col: str = "timestamp",
    date_now: Optional[date] = None,
    churn_days: int = 90,
    active_min_sessions: int = 5,
    active_recency_days: int = 30
) -> pd.DataFrame:
    """
    Присваивает activity_segment каждому событию/строке:
      - New: если user_first_visit_date == date_now (сравнение по дате)
      - Returning: базовая
      - Churn-risk: last_visit_date < date_now - churn_days
      - Active: n_sessions >= active_min_sessions and last_visit_date >= date_now - active_recency_days

    Возвращает копию df с колонками: first_visit_day, last_visit_date, n_sessions, activity_segment.
    """
    df_local = df.copy()
    if date_now is None:
        date_now = datetime.utcnow().date()

    # гарантируем, что first_visit_col является datetime
    if first_visit_col not in df_local.columns and "userid" in df_local.columns and ts_col in df_local.columns:
        # вычисляем first_visit если отсутствует
        first_visit = df_local.groupby("userid")[ts_col].transform("min")
        df_local[first_visit_col] = first_visit

    if first_visit_col in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local[first_visit_col]):
        df_local[first_visit_col] = pd.to_datetime(df_local[first_visit_col], errors="coerce")
    # сравнивать по дню
    df_local["first_visit_day"] = df_local[first_visit_col].dt.date

    # ensure timestamp and date fields
    if ts_col in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local[ts_col]):
        df_local[ts_col] = pd.to_datetime(df_local[ts_col], errors="coerce")
    df_local["date"] = df_local[ts_col].dt.date

    # last_visit_date per user
    df_local["last_visit_date"] = df_local.groupby("userid")["date"].transform("max")
    df_local["n_sessions"] = df_local.groupby("userid")["sessionid"].transform("nunique")

    # default
    df_local["activity_segment"] = "Returning"
    # new
    df_local.loc[df_local["first_visit_day"] == date_now, "activity_segment"] = "New"
    # churn risk
    df_local.loc[df_local["last_visit_date"] < (date_now - timedelta(days=churn_days)), "activity_segment"] = "Churn-risk"
    # active
    df_local.loc[
        (df_local["n_sessions"] >= active_min_sessions) & (df_local["last_visit_date"] >= (date_now - timedelta(days=active_recency_days))),
        "activity_segment"
    ] = "Active"
    return df_local


def cohort_month_retention(df: pd.DataFrame, userid_col: str = "userid", ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Базовый расчет удержания по когортам по месяцу начала:
      - определяем first_timestamp для каждого пользователя
      - cohort_month = first_timestamp.to_period('M').to_timestamp()
      - cohort_lifetime_days = (timestamp - cohort_month).days
      - считаем уникальных users по (cohort_month, cohort_lifetime_days)
      - добавляем cohort_size

    Возвращаем DataFrame с колонками:
      ['cohort_month', 'cohort_lifetime_days', 'retained_users', 'cohort_size']
    """
    df_local = df.copy()
    if ts_col not in df_local.columns:
        raise ValueError(f"{ts_col} not present in df")
    if not pd.api.types.is_datetime64_any_dtype(df_local[ts_col]):
        df_local[ts_col] = pd.to_datetime(df_local[ts_col], errors="coerce")

    first_ts = df_local.groupby(userid_col)[ts_col].min().rename("first_timestamp").reset_index()
    df_local = df_local.merge(first_ts, on=userid_col, how="left")
    df_local["cohort_month"] = df_local["first_timestamp"].dt.to_period("M").dt.to_timestamp()
    df_local["cohort_lifetime_days"] = (df_local[ts_col] - df_local["cohort_month"]).dt.days

    retained = (
        df_local.groupby(["cohort_month", "cohort_lifetime_days"])[userid_col]
        .nunique()
        .reset_index()
        .rename(columns={userid_col: "retained_users"})
    )

    cohort_sizes = (
        first_ts.groupby(first_ts["first_timestamp"].dt.to_period("M").dt.to_timestamp())  # index is timestamp
        .size()
        .rename("cohort_size")
        .reset_index()
        .rename(columns={"first_timestamp": "cohort_month"})
    )

    retention = retained.merge(cohort_sizes, on="cohort_month", how="left")
    return retention
