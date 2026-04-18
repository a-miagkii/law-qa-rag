from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd


def norm_text(x) -> str:
    """
    Normalize text for stable comparisons:
    - NaN -> ""
    - Unicode normalization
    - remove non-breaking spaces
    - collapse repeated whitespace
    - lowercase
    """
    if pd.isna(x):
        return ""
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = x.replace("\xa0", " ")
    x = " ".join(x.split())
    return x.strip().lower()


def add_missing_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    return df


def build_fz_manifest(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "file_path",
        "source_nd",
        "doc_type",
        "title",
        "issued_by",
        "doc_author_normal_form",
        "doc_number",
        "doc_date_raw",
        "doc_date_iso",
        "status",
        "actual_datetime_human",
        "is_widely_used",
        "classifier",
        "keywords_count",
        "text_len_chars",
        "has_text",
        "parse_error",
    ]
    df = add_missing_columns(df.copy(), required_cols)

    # Normalized helper columns
    for col in ["doc_type", "title", "issued_by", "status", "doc_number"]:
        df[f"{col}_norm"] = df[col].apply(norm_text)

    # Main rules:
    # 1) explicit doc_type == "Федеральный закон"
    rule_doc_type = df["doc_type_norm"] == "федеральный закон"

    # 2) doc_type missing, but issued_by says "Федеральный закон"
    rule_issued_by_fallback = (
        (df["doc_type_norm"] == "")
        & (df["issued_by_norm"] == "федеральный закон")
    )

    # 3) title begins with "Федеральный закон"
    rule_title_prefix = df["title_norm"].str.startswith("федеральный закон", na=False)

    # NOTE:
    # We intentionally do NOT use doc_number contains "ФЗ" as a primary rule,
    # because codes can also have numbers like "146-ФЗ".
    df["fz_by_doc_type"] = rule_doc_type
    df["fz_by_issued_by_fallback"] = rule_issued_by_fallback
    df["fz_by_title_prefix"] = rule_title_prefix

    df["is_federal_law"] = (
        df["fz_by_doc_type"]
        | df["fz_by_issued_by_fallback"]
        | df["fz_by_title_prefix"]
    )

    def detect_reason(row) -> str:
        reasons = []
        if row["fz_by_doc_type"]:
            reasons.append("doc_type")
        if row["fz_by_issued_by_fallback"]:
            reasons.append("issued_by_fallback")
        if row["fz_by_title_prefix"]:
            reasons.append("title_prefix")
        return "|".join(reasons)

    df["federal_law_reason"] = df.apply(detect_reason, axis=1)

    # Optional helper flag
    df["is_code"] = df["doc_type_norm"] == "кодекс"

    fz_df = df[df["is_federal_law"]].copy()

    # Sort for convenience
    sort_cols = [c for c in ["doc_date_iso", "title"] if c in fz_df.columns]
    if sort_cols:
        fz_df = fz_df.sort_values(sort_cols, ascending=[True, True])

    preferred_columns = [
        "file_path",
        "source_nd",
        "doc_type",
        "title",
        "issued_by",
        "doc_author_normal_form",
        "doc_number",
        "doc_date_raw",
        "doc_date_iso",
        "status",
        "actual_datetime_human",
        "is_widely_used",
        "classifier",
        "keywords_count",
        "text_len_chars",
        "has_text",
        "parse_error",
        "is_federal_law",
        "federal_law_reason",
        "fz_by_doc_type",
        "fz_by_issued_by_fallback",
        "fz_by_title_prefix",
        "is_code",
        "doc_type_norm",
        "issued_by_norm",
        "title_norm",
        "status_norm",
        "doc_number_norm",
    ]

    existing_columns = [c for c in preferred_columns if c in fz_df.columns]
    other_columns = [c for c in fz_df.columns if c not in existing_columns]

    return fz_df[existing_columns + other_columns]


def main():
    parser = argparse.ArgumentParser(description="Build fz_manifest.csv from all_docs_manifest.csv")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to all_docs_manifest.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output fz_manifest.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    fz_df = build_fz_manifest(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fz_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")
    print(f"Total input rows: {len(df)}")
    print(f"Federal law rows: {len(fz_df)}")
    print()
    print("Breakdown:")
    print(f"  by doc_type: {int(fz_df['fz_by_doc_type'].sum())}")
    print(f"  by issued_by fallback: {int(fz_df['fz_by_issued_by_fallback'].sum())}")
    print(f"  by title prefix: {int(fz_df['fz_by_title_prefix'].sum())}")


if __name__ == "__main__":
    main()