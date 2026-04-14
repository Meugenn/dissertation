from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .real_market import ObservedMarket
from .synthetic_market import TemporalEdge


def _standardize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((matrix - mean) / std).astype(np.float64)


def _top_k_dummies(series: pd.Series, prefix: str, top_k: int = 6) -> pd.DataFrame:
    normalized = series.fillna("missing").astype(str)
    top_values = normalized.value_counts().head(top_k).index
    clipped = normalized.where(normalized.isin(top_values), other="other")
    return pd.get_dummies(clipped, prefix=prefix, dtype=float)


def load_hm_local_market(
    data_dir: Path,
    *,
    max_rows: int | None = 300_000,
    min_customer_transactions: int = 3,
    min_article_transactions: int = 3,
    train_fraction: float = 0.8,
) -> ObservedMarket:
    customers_path = data_dir / "customers.csv"
    articles_path = data_dir / "articles.csv"
    transactions_path = data_dir / "transactions_train.csv"
    for path in (customers_path, articles_path, transactions_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected H&M file at {path}")

    transactions = pd.read_csv(
        transactions_path,
        usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        nrows=max_rows,
    )
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], utc=True)
    transactions = transactions.dropna(subset=["customer_id", "article_id", "t_dat"])

    customer_counts = transactions["customer_id"].value_counts()
    article_counts = transactions["article_id"].value_counts()
    keep_customers = set(customer_counts[customer_counts >= min_customer_transactions].index.astype(str))
    keep_articles = set(article_counts[article_counts >= min_article_transactions].index.astype(str))

    transactions["customer_id"] = transactions["customer_id"].astype(str)
    transactions["article_id"] = transactions["article_id"].astype(str)
    transactions = transactions[
        transactions["customer_id"].isin(keep_customers) & transactions["article_id"].isin(keep_articles)
    ].copy()
    if transactions.empty:
        raise ValueError("No H&M transactions remained after filtering; lower the minimum transaction thresholds or increase max_rows.")

    customers = pd.read_csv(customers_path)
    customers["customer_id"] = customers["customer_id"].astype(str)
    customers = customers[customers["customer_id"].isin(set(transactions["customer_id"]))].copy()

    articles = pd.read_csv(articles_path)
    articles["article_id"] = articles["article_id"].astype(str)
    articles = articles[articles["article_id"].isin(set(transactions["article_id"]))].copy()

    customer_numeric = customers[["age"]].copy()
    customer_numeric["age"] = customer_numeric["age"].fillna(customer_numeric["age"].median())
    for column in ["FN", "Active"]:
        if column in customers:
            customer_numeric[column] = customers[column].fillna(0).astype(float)
    customer_cats = pd.concat(
        [
            _top_k_dummies(customers.get("club_member_status", pd.Series(index=customers.index, dtype=str)), "club", top_k=4),
            _top_k_dummies(customers.get("fashion_news_frequency", pd.Series(index=customers.index, dtype=str)), "news", top_k=4),
        ],
        axis=1,
    )
    customer_features = pd.concat([customer_numeric.reset_index(drop=True), customer_cats.reset_index(drop=True)], axis=1)

    article_numeric_columns = [
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "department_no",
        "section_no",
        "garment_group_no",
    ]
    article_numeric = articles[[column for column in article_numeric_columns if column in articles]].fillna(0).astype(float)
    article_cats = pd.concat(
        [
            _top_k_dummies(articles.get("index_group_name", pd.Series(index=articles.index, dtype=str)), "index_group", top_k=4),
            _top_k_dummies(articles.get("product_group_name", pd.Series(index=articles.index, dtype=str)), "product_group", top_k=4),
        ],
        axis=1,
    )
    article_features = pd.concat([article_numeric.reset_index(drop=True), article_cats.reset_index(drop=True)], axis=1)

    buyer_ids = tuple(customers["customer_id"].tolist())
    seller_ids = tuple(articles["article_id"].tolist())
    buyer_index = {customer_id: idx for idx, customer_id in enumerate(buyer_ids)}
    seller_index = {article_id: idx for idx, article_id in enumerate(seller_ids)}

    transactions = transactions.sort_values("t_dat").reset_index(drop=True)
    edges_with_time: list[tuple[float, TemporalEdge]] = []
    for row in transactions.itertuples(index=False):
        edge_type = 0 if int(getattr(row, "sales_channel_id")) == 1 else 1
        timestamp = pd.Timestamp(getattr(row, "t_dat")).timestamp()
        edges_with_time.append(
            (
                timestamp,
                TemporalEdge(
                    buyer=buyer_index[str(getattr(row, "customer_id"))],
                    seller=seller_index[str(getattr(row, "article_id"))],
                    edge_type=edge_type,
                    time=timestamp,
                ),
            )
        )

    split_idx = max(1, min(len(edges_with_time) - 1, int(len(edges_with_time) * train_fraction)))
    train_edges = tuple(edge for _, edge in edges_with_time[:split_idx])
    eval_edges = tuple(edge for _, edge in edges_with_time[split_idx:])
    train_cutoff = edges_with_time[split_idx - 1][0]
    total_horizon = edges_with_time[-1][0]

    metadata = {
        "raw_transactions": int(len(pd.read_csv(transactions_path, usecols=["customer_id"], nrows=max_rows))),
        "filtered_transactions": int(len(transactions)),
        "max_rows": max_rows,
    }

    return ObservedMarket(
        buyer_ids=buyer_ids,
        seller_ids=seller_ids,
        buyer_features=_standardize(customer_features.to_numpy(dtype=np.float64)),
        seller_features=_standardize(article_features.to_numpy(dtype=np.float64)),
        train_edges=train_edges,
        eval_edges=eval_edges,
        edge_type_names=("channel_1", "channel_2"),
        edge_type_weights=np.array([1.0, 1.0], dtype=np.float64),
        train_cutoff=train_cutoff,
        total_horizon=total_horizon,
        source_name="hm_local",
        metadata=metadata,
    )
