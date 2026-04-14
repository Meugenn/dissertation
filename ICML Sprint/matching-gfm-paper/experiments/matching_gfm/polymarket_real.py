from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from .real_market import ObservedMarket
from .synthetic_market import TemporalEdge


GAMMA_API_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def _request_json(base_url: str, path: str, params: dict[str, Any]) -> Any:
    query = urlencode(params, doseq=True)
    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{query}"
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_json_field(raw: str | list[Any] | None) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _standardize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((matrix - mean) / std).astype(np.float64)


def _top_categories(categories: list[str], top_k: int) -> list[str]:
    counts = Counter(cat for cat in categories if cat)
    return [name for name, _ in counts.most_common(top_k)]


def _trade_timestamp(raw: dict[str, Any]) -> float:
    timestamp = raw.get("timestamp")
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    if isinstance(timestamp, str):
        try:
            return float(timestamp)
        except ValueError:
            try:
                cleaned = timestamp.replace("Z", "+00:00")
                return datetime.fromisoformat(cleaned).timestamp()
            except ValueError:
                pass
    return float(datetime.now(timezone.utc).timestamp())


@dataclass(frozen=True)
class PolymarketSnapshot:
    markets: list[dict[str, Any]]
    trades: list[dict[str, Any]]


def fetch_polymarket_snapshot(
    *,
    max_event_pages: int = 3,
    max_trade_pages: int = 6,
    event_page_size: int = 100,
    trade_page_size: int = 500,
) -> PolymarketSnapshot:
    events: list[dict[str, Any]] = []
    offset = 0
    for _ in range(max_event_pages):
        page = _request_json(
            GAMMA_API_URL,
            "/events",
            {
                "active": "true",
                "closed": "false",
                "limit": event_page_size,
                "offset": offset,
                "order": "id",
                "ascending": "true",
            },
        )
        if not page:
            break
        events.extend(page)
        if len(page) < event_page_size:
            break
        offset += event_page_size

    markets: list[dict[str, Any]] = []
    for event in events:
        for market in event.get("markets") or []:
            market = dict(market)
            market["_event_title"] = event.get("title", "")
            market["_event_slug"] = event.get("slug", "")
            tags = event.get("tags") or []
            market["_category"] = tags[0]["label"] if tags and isinstance(tags[0], dict) and "label" in tags[0] else ""
            markets.append(market)

    trades: list[dict[str, Any]] = []
    offset = 0
    for _ in range(max_trade_pages):
        page = _request_json(
            DATA_API_URL,
            "/trades",
            {
                "limit": trade_page_size,
                "offset": offset,
            },
        )
        if not page:
            break
        trades.extend(page)
        if len(page) < trade_page_size:
            break
        offset += trade_page_size

    return PolymarketSnapshot(markets=markets, trades=trades)


def load_polymarket_market(
    *,
    max_event_pages: int = 3,
    max_trade_pages: int = 6,
    min_wallet_trades: int = 3,
    min_market_trades: int = 3,
    max_wallets: int = 250,
    max_markets: int = 250,
    top_category_count: int = 6,
    train_fraction: float = 0.8,
    snapshot_dir: str | None = None,
) -> ObservedMarket:
    snapshot = fetch_polymarket_snapshot(
        max_event_pages=max_event_pages,
        max_trade_pages=max_trade_pages,
    )

    market_by_condition: dict[str, dict[str, Any]] = {}
    for market in snapshot.markets:
        condition_id = market.get("conditionId") or market.get("condition_id")
        if condition_id:
            market_by_condition[condition_id] = market

    filtered_trades = [trade for trade in snapshot.trades if (trade.get("conditionId") or "") in market_by_condition]
    wallet_counts = Counter(trade.get("proxyWallet", "") for trade in filtered_trades if trade.get("proxyWallet"))
    market_counts = Counter(trade.get("conditionId", "") for trade in filtered_trades if trade.get("conditionId"))

    top_wallets = {
        wallet
        for wallet, _ in wallet_counts.most_common(max_wallets)
        if wallet and wallet_counts[wallet] >= min_wallet_trades
    }
    top_markets = {
        condition_id
        for condition_id, _ in market_counts.most_common(max_markets)
        if condition_id and market_counts[condition_id] >= min_market_trades
    }

    trades = [
        trade
        for trade in filtered_trades
        if trade.get("proxyWallet", "") in top_wallets and trade.get("conditionId", "") in top_markets
    ]
    if not trades:
        raise ValueError("No Polymarket trades remained after filtering; try increasing max_trade_pages or lowering minimum thresholds.")

    buyers = sorted({trade["proxyWallet"] for trade in trades})
    sellers = sorted({trade["conditionId"] for trade in trades})
    buyer_index = {wallet: idx for idx, wallet in enumerate(buyers)}
    seller_index = {condition_id: idx for idx, condition_id in enumerate(sellers)}

    categories = [market_by_condition[seller].get("_category", "") for seller in sellers]
    top_categories = _top_categories(categories, top_category_count)
    category_to_idx = {name: idx for idx, name in enumerate(top_categories)}

    wallet_stats = defaultdict(lambda: {"count": 0, "volume": 0.0, "buy": 0, "sell": 0, "markets": set(), "prices": [], "times": []})
    for trade in trades:
        wallet = trade["proxyWallet"]
        stats = wallet_stats[wallet]
        size = float(trade.get("size") or 0.0)
        price = float(trade.get("price") or 0.0)
        stats["count"] += 1
        stats["volume"] += size * max(price, 1e-9)
        stats["buy"] += int(str(trade.get("side", "")).upper() == "BUY")
        stats["sell"] += int(str(trade.get("side", "")).upper() != "BUY")
        stats["markets"].add(trade["conditionId"])
        stats["prices"].append(price)
        stats["times"].append(_trade_timestamp(trade))

    buyer_rows: list[np.ndarray] = []
    for wallet in buyers:
        stats = wallet_stats[wallet]
        exposure = np.zeros(len(top_categories), dtype=np.float64)
        for trade in trades:
            if trade["proxyWallet"] != wallet:
                continue
            category = market_by_condition[trade["conditionId"]].get("_category", "")
            if category in category_to_idx:
                exposure[category_to_idx[category]] += 1.0
        trade_count = max(stats["count"], 1)
        buyer_rows.append(
            np.concatenate(
                [
                    np.array(
                        [
                            np.log1p(stats["count"]),
                            np.log1p(stats["volume"]),
                            stats["buy"] / trade_count,
                            stats["sell"] / trade_count,
                            len(stats["markets"]),
                            float(np.mean(stats["prices"])) if stats["prices"] else 0.0,
                            float(np.std(stats["prices"])) if stats["prices"] else 0.0,
                            float(max(stats["times"]) - min(stats["times"])) if len(stats["times"]) > 1 else 0.0,
                        ],
                        dtype=np.float64,
                    ),
                    exposure,
                ]
            )
        )
    buyer_features = _standardize(np.vstack(buyer_rows))

    seller_rows: list[np.ndarray] = []
    for condition_id in sellers:
        market = market_by_condition[condition_id]
        category = market.get("_category", "")
        category_vector = np.zeros(len(top_categories), dtype=np.float64)
        if category in category_to_idx:
            category_vector[category_to_idx[category]] = 1.0
        outcome_prices = [float(x) for x in _parse_json_field(market.get("outcomePrices")) if str(x) not in {"", "null"}]
        seller_rows.append(
            np.concatenate(
                [
                    np.array(
                        [
                            np.log1p(float(market.get("volumeNum") or market.get("volume") or 0.0)),
                            np.log1p(float(market.get("liquidityNum") or market.get("liquidity") or 0.0)),
                            np.log1p(float(market.get("volume24hr") or 0.0)),
                            float(market.get("competitive") or 0.0),
                            float(market.get("commentCount") or 0.0),
                            float(bool(market.get("active"))),
                            float(bool(market.get("closed"))),
                            float(bool(market.get("enableOrderBook"))),
                            len(_parse_json_field(market.get("outcomes"))),
                            float(np.mean(outcome_prices)) if outcome_prices else 0.0,
                        ],
                        dtype=np.float64,
                    ),
                    category_vector,
                ]
            )
        )
    seller_features = _standardize(np.vstack(seller_rows))

    edges_with_time: list[tuple[float, TemporalEdge]] = []
    for trade in trades:
        timestamp = _trade_timestamp(trade)
        side = str(trade.get("side", "")).upper()
        edge_type = 0 if side == "BUY" else 1
        edges_with_time.append(
            (
                timestamp,
                TemporalEdge(
                    buyer=buyer_index[trade["proxyWallet"]],
                    seller=seller_index[trade["conditionId"]],
                    edge_type=edge_type,
                    time=timestamp,
                ),
            )
        )
    edges_with_time.sort(key=lambda item: item[0])
    split_idx = max(1, min(len(edges_with_time) - 1, int(len(edges_with_time) * train_fraction)))
    train_edges = tuple(edge for _, edge in edges_with_time[:split_idx])
    eval_edges = tuple(edge for _, edge in edges_with_time[split_idx:])
    train_cutoff = edges_with_time[split_idx - 1][0]
    total_horizon = edges_with_time[-1][0]

    metadata = {
        "num_raw_markets": len(snapshot.markets),
        "num_raw_trades": len(snapshot.trades),
        "num_filtered_trades": len(trades),
        "top_categories": top_categories,
        "min_wallet_trades": min_wallet_trades,
        "min_market_trades": min_market_trades,
    }

    market = ObservedMarket(
        buyer_ids=tuple(buyers),
        seller_ids=tuple(sellers),
        buyer_features=buyer_features,
        seller_features=seller_features,
        train_edges=train_edges,
        eval_edges=eval_edges,
        edge_type_names=("buy", "sell"),
        edge_type_weights=np.array([1.0, 1.0], dtype=np.float64),
        train_cutoff=train_cutoff,
        total_horizon=total_horizon,
        source_name="polymarket",
        metadata=metadata,
    )

    if snapshot_dir is not None:
        output_dir = Path(snapshot_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        market.write_snapshot(output_dir)

    return market
