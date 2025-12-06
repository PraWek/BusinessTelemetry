#!/usr/bin/env python3
"""
main.py

Демонстрация использования функций из:
 - data_cleaning.py
 - feature_engineering.py
 - metrics.py
 - cohorts.py

Usage:
    python main.py --input dataset_telemetry.csv --output ./output
"""
import os
import argparse
import pprint

import pandas as pd

# Модули рефакторинга — предполагается, что они находятся рядом с main.py
import data_cleaning as dc
import feature_engineering as fe
import metrics as mtr
import cohorts as coh


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_basic_event_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт cart_value и checkout_value на основе колонки 'value' и 'action',
    аналогично вашему ноутбуку.
    """
    df = df.copy()
    if "value" in df.columns:
        df["cart_value"] = df["value"].where(df["action"] == "cart", 0)
        df["checkout_value"] = df["value"].where(df["action"] == "checkout", 0)
    else:
        df["cart_value"] = 0
        df["checkout_value"] = 0
    return df


def main(input_path: str, output_dir: str):
    print(">>> Starting demo main.py")
    ensure_output_dir(output_dir)

    # 1) Read & basic cleaning
    print(f">>> Reading file: {input_path}")
    df = dc.read_telemetry_csv(input_path)
    print("Initial shape:", df.shape)

    # 2) Fill category NaNs
    df = dc.safe_fill_category(df, col="category", fill_value="unknown")

    # 3) Ensure timestamp parsing and date col
    df = dc.parse_timestamps(df, ts_col="timestamp")
    print("Timestamp parsed. Sample:")
    print(df[["userid", "sessionid", "timestamp", "date", "action"]].head(3).to_string(index=False))

    # 4) session aggregates (session_duration at session level and merged back)
    df, sessions = dc.compute_session_aggregates(df, session_col="sessionid", ts_col="timestamp", keep_events=True)
    print("Sessions computed:", sessions.shape)
    print(sessions.head(3).to_string(index=False))

    # 5) derive cart/checkout value columns (if they don't exist)
    df = prepare_basic_event_values(df)

    # 6) add session step numbers & shifts
    df = fe.compute_session_step_number(df, session_col="sessionid", ts_col="timestamp")
    df = fe.add_session_shifts(df, session_col="sessionid", ts_col="timestamp")
    print("Added session step numbers and session-aware shifts.")

    # 7) compute basket size and avg time between cart and checkout
    df = fe.compute_basket_size_and_avg_time(df, session_col="sessionid", ts_col="timestamp")
    print("Computed basket_size and avg_time_between_cart_and_checkout. Sample:")
    print(df[["sessionid", "action", "product_to_cart", "basket_size", "avg_time_between_cart_and_checkout"]].head(10).to_string(index=False))

    # 8) product->cart transitions (list)
    transitions = fe.product_to_cart_transitions(df, session_col="sessionid", ts_col="timestamp")
    print("Product -> Cart transitions (sample):")
    print(transitions.head(10).to_string(index=False))

    # 9) define funnel steps (order matters)
    steps = ["search", "product", "category", "mainpage", "cart", "checkout", "confirmation"]

    # 10) compute funnel (unique users per session-step in order)
    funnel = mtr.compute_funnel(df, steps=steps, session_col="sessionid", userid_col="userid", ts_col="timestamp")
    print("Funnel results:")
    print(funnel.to_string(index=False))

    # 11) KPI by date
    kpis = mtr.compute_kpis_by_date(df, date_col="date", orders_action="checkout", confirmation_col=None)
    print("KPIs by date (sample):")
    print(kpis.head(10).to_string(index=False))

    # 12) Sankey transitions (action->next_action) inside sessions
    # create a step_map similar to ваш ноутбук
    step_map = {action: i for i, action in enumerate(steps)}
    sankey = mtr.compute_sankey_transitions(df, step_map=step_map, session_col="sessionid", ts_col="timestamp", require_step_increase=True)
    print("Sankey transitions (sample):")
    print(sankey.head(10).to_string(index=False))

    # 13) daily conversions across steps
    conversion_daily = mtr.compute_conversion_daily(df, steps=steps, date_col="date")
    print("Daily conversions (sample):")
    print(conversion_daily.head(10).to_string(index=False))

    # 14) label user activity (New / Returning / Churn-risk / Active)
    # this function will compute first_visit if it's not in df
    labeled = coh.label_user_activity_segment(df, first_visit_col="user_first_visit_date", ts_col="timestamp")
    print("Activity segments sample:")
    print(labeled[["userid", "first_visit_day", "last_visit_date", "n_sessions", "activity_segment"]].drop_duplicates(subset=["userid"]).head(10).to_string(index=False))

    # 15) cohort month retention
    retention = coh.cohort_month_retention(df, userid_col="userid", ts_col="timestamp")
    print("Retention (sample):")
    print(retention.head(10).to_string(index=False))

    # 16) Save some outputs
    print(f">>> Saving outputs to {output_dir}")
    out_files = {
        "cleaned_events.csv": df,
        "sessions.csv": sessions,
        "transitions_product_to_cart.csv": transitions,
        "funnel.csv": funnel,
        "kpis_by_date.csv": kpis,
        "sankey.csv": sankey,
        "conversion_daily.csv": conversion_daily,
        "activity_labeled.csv": labeled,
        "cohort_retention.csv": retention
    }
    for fname, obj in out_files.items():
        path = os.path.join(output_dir, fname)
        try:
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=False)
            else:
                # fallback: pretty print non-dataframes to text
                with open(path, "w", encoding="utf-8") as f:
                    f.write(str(obj))
            print(f"  saved {fname}")
        except Exception as e:
            print(f"  failed to save {fname}: {e}")

    print(">>> Demo finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo runner for refactored telemetry modules")
    parser.add_argument("--input", "-i", type=str, default="dataset_telemetry.csv", help="Path to telemetry CSV")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Output directory")
    args = parser.parse_args()

    main(input_path=args.input, output_dir=args.output)
