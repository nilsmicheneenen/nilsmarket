# streamlit_cross_exchange_prediction_market_arbitrage_finder.py
"""
Prediction Market Cross-Exchange Arbitrage Finder
Platforms: PredictIt, Polymarket, (optional) Kalshi

Features
- Fetches public markets (Gamma for Polymarket, PredictIt API; Kalshi optional with creds)
- Robust Polymarket price fetching (doc-compliant) with fallbacks & chunking
- Fuzzy cross-venue matching (RapidFuzz if available) with union-find clustering
- Fee-aware arbitrage detection (buy load %, payout haircut %, settlement $)
- Cross-exchange enforcement (ON by default)
- Budget-aware sizing + CSV exports
- Full table of all fetched markets & grouped view (Yes/No columns)

Run
----
1) Save this file.
2) pip install -r requirements.txt  (see bottom)
3) streamlit run streamlit_cross_exchange_prediction_market_arbitrage_finder.py

DISCLAIMER: Educational only. Obey venue/jurisdiction rules and confirm fees/data before acting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from difflib import SequenceMatcher

# ==========================
# Streamlit setup
# ==========================
st.set_page_config(page_title="Prediction Market Arbitrage Finder", layout="wide")

# ==========================
# Helpers
# ==========================
@st.cache_data(ttl=60)
def http_get_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        st.warning(f"GET {url} → HTTP {r.status_code}")
        return None
    except Exception as e:
        st.warning(f"GET {url} failed: {e}")
        return None

# ==========================
# Fee Model
# ==========================
@dataclass
class FeeModel:
    buy_load: float = 0.0          # extra % on buy: p * (1+buy_load)
    payout_haircut: float = 0.0    # % haircut on $1 payout
    settlement_fee: float = 0.0    # flat $ fee on winning leg (approx)

    def effective_buy_price(self, p: float) -> float:
        return p * (1.0 + self.buy_load)

    def effective_payout(self, gross: float) -> float:
        return max(0.0, gross * (1.0 - self.payout_haircut) - self.settlement_fee)

# ==========================
# Data structures
# ==========================
@dataclass
class Quote:
    exchange: str
    market_id: str
    market_name: str
    outcome: str      # "Yes" or "No"
    best_buy_cost: Optional[float]
    symbol: Optional[str] = None
    url: Optional[str] = None

@dataclass
class MatchedMarket:
    key: str
    titles: Dict[str, str]
    quotes: List[Quote]

@dataclass
class Arb:
    group_key: str
    title_hint: str
    buy_yes: Quote
    buy_no: Quote
    total_cost: float
    margin: float

    def max_pairs(self, budget: float) -> int:
        return int(budget // self.total_cost) if self.total_cost > 0 else 0

    def per_pair_profit(self) -> float:
        return max(0.0, 1.0 - self.total_cost)

# ==========================
# Exchange connectors
# ==========================
@st.cache_data(ttl=60)
def fetch_predictit() -> List[Quote]:
    url = "https://www.predictit.org/api/marketdata/all/"
    data = http_get_json(url)
    quotes: List[Quote] = []
    if not data:
        return quotes
    for m in data.get("markets", []):
        market_name = m.get("name", "")
        base_url = m.get("url")
        for c in m.get("contracts", []):
            cid = str(c.get("id"))
            y = c.get("bestBuyYesCost")
            n = c.get("bestBuyNoCost")
            if y is not None:
                quotes.append(Quote("PredictIt", cid, market_name, "Yes", float(y), url=base_url))
            if n is not None:
                quotes.append(Quote("PredictIt", cid, market_name, "No", float(n), url=base_url))
    return quotes

@st.cache_data(ttl=60)
def fetch_polymarket() -> List[Quote]:
    """Polymarket full-scan: up to 3x1000 pages from Gamma, de-duped, then CLOB prices."""
    gamma = "https://gamma-api.polymarket.com"
    clob = "https://clob.polymarket.com"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "ArbFinder/1.6 (+streamlit)",
        "Accept": "application/json",
    })

    # -------- 1) Paged discovery (3 pages × 1000) with de-dupe --------
    PAGE_LIMIT = 1000
    PAGE_COUNT = 6
    markets_map: Dict[str, dict] = {}
    total_fetched = 0

    for page in range(PAGE_COUNT):
        try:
            r = session.get(
                f"{gamma}/markets",
                params={"limit": PAGE_LIMIT, "offset": page * PAGE_LIMIT, "closed": False},
                timeout=25,
            )
            r.raise_for_status()
            js = r.json()
            page_markets = js if isinstance(js, list) else js.get("data", []) or []
        except Exception as e:
            st.warning(f"Polymarket Gamma /markets page {page+1} error: {e}")
            page_markets = []

        for m in page_markets:
            # Build a stable unique key across possible shapes
            uniq_key = str(
                m.get("questionID")
                or m.get("conditionId")
                or m.get("id")
                or m.get("slug")
                or json.dumps(m.get("tokens") or m.get("outcomes") or m, sort_keys=True)[:64]
            )
            if uniq_key not in markets_map:
                markets_map[uniq_key] = m
        total_fetched += len(page_markets)

    markets = list(markets_map.values())
    st.caption(
        f"Polymarket full scan — pages: {PAGE_COUNT}, per-page: {PAGE_LIMIT}, "
        f"fetched: {total_fetched}, unique after de-dupe: {len(markets)}"
    )

    # -------- helper to read token IDs regardless of shape --------
    def as_ids(val) -> List[str]:
        if val is None: return []
        if isinstance(val, list): return [str(x) for x in val if x is not None]
        if isinstance(val, dict):
            out = []
            for k in ("YES","yes","Yes","NO","no","No","0","1"):
                if k in val and val[k] is not None:
                    out.append(str(val[k]))
            return out
        if isinstance(val, str):
            try: return as_ids(json.loads(val))
            except Exception:
                return [s.strip("'\" ") for s in val.strip("[]").split(",") if s.strip()]
        return []

    # -------- 2) Build price request payloads --------
    price_params: List[Dict[str, str]] = []
    token_meta: List[Tuple[str, str, str, str, Optional[str]]] = []
    inactive = non_clob = no_ids = 0

    for m in markets:
        if not m.get("active", True):
            inactive += 1
            continue
        if m.get("enableOrderBook") is False:
            non_clob += 1
            continue

        q = m.get("question") or m.get("title") or m.get("name") or m.get("slug") or "(untitled)"
        slug = m.get("slug")
        mid = str(m.get("questionID") or m.get("conditionId") or m.get("id") or slug or "?")

        ids = as_ids(m.get("clobTokenIds") or m.get("tokenIds"))
        if len(ids) != 2:
            outs = m.get("outcomes")
            if isinstance(outs, list) and len(outs) == 2:
                tmp = []
                for o in outs:
                    tid = None
                    if isinstance(o, dict):
                        tid = o.get("clobTokenId") or o.get("tokenId") or o.get("id")
                    tmp.append(str(tid) if tid is not None else "")
                ids = tmp

        if len(ids) != 2 or not ids[0] or not ids[1]:
            no_ids += 1
            continue

        labels = m.get("shortOutcomes")
        if not (isinstance(labels, list) and len(labels) == 2):
            labels = ["Yes", "No"]

        for idx, tid in enumerate(ids[:2]):
            outcome = str(labels[idx]).strip().title() if idx < len(labels) else ("Yes" if idx == 0 else "No")
            if outcome not in ("Yes", "No"):
                outcome = "Yes" if idx == 0 else "No"
            price_params.append({"token_id": str(tid), "side": "BUY"})
            token_meta.append((mid, q, outcome, str(tid), slug))

    st.caption(
        f"Polymarket filter — inactive: {inactive}, non-CLOB: {non_clob}, "
        f"missing IDs: {no_ids}, tokens to price: {len(price_params)}"
    )

    if not price_params:
        return []

    # -------- 3) Fetch prices in chunks, try multiple payload shapes --------
    def fetch_prices_batch(batch: List[Dict[str,str]]) -> Dict[str, Any]:
        try:
            r = session.post(f"{clob}/prices", json={"params": batch}, timeout=25)
            if r.status_code == 200:
                return r.json() or {}
            r = session.post(f"{clob}/prices", json=batch, timeout=25)
            if r.status_code == 200:
                return r.json() or {}
            token_ids = [bp["token_id"] for bp in batch]
            r = session.post(f"{clob}/prices", json={"token_ids": token_ids, "side": "BUY"}, timeout=25)
            if r.status_code == 200:
                return r.json() or {}
            return {}
        except Exception:
            return {}

    # de-dup token_ids
    seen_tids = set()
    dedup_params = []
    for p in price_params:
        tid = str(p.get("token_id"))
        if tid and tid not in seen_tids and tid.lower() != "none" and tid != "0":
            seen_tids.add(tid)
            dedup_params.append({"token_id": tid, "side": "BUY"})

    CHUNK = 200
    price_map_all: Dict[str, Any] = {}
    for i in range(0, len(dedup_params), CHUNK):
        price_map_all.update(fetch_prices_batch(dedup_params[i:i+CHUNK]))

    if not price_map_all:
        st.warning("Polymarket: /prices returned empty (possible rate-limit or schema change).")
        return []

    # -------- 4) Build quotes --------
    quotes: List[Quote] = []
    for (mid, q, outcome, tid, slug) in token_meta:
        entry = price_map_all.get(tid)
        if entry is None:
            continue
        buy_price = None
        if isinstance(entry, dict):
            buy_price = entry.get("BUY") or entry.get("buy")
        elif isinstance(entry, (int, float, str)):
            try:
                buy_price = float(entry)
            except Exception:
                buy_price = None
        if buy_price is None:
            continue
        try:
            p = float(buy_price)
        except Exception:
            continue
        url_hint = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com/"
        quotes.append(Quote("Polymarket", mid, q, outcome, p, url=url_hint))

    return quotes


# ==========================
# Keyword-driven Polymarket fetch
# ==========================
@st.cache_data(ttl=60)
def fetch_polymarket_by_keywords(candidate_titles: List[str], extra_keywords: List[str],
                                 similarity: float = 0.8, min_shared: int = 2, limit_markets: int = 500,
                                 fallback_full_if_empty: bool = True) -> List[Quote]:
    """Fetch a *subset* of Polymarket by matching keywords from other venues' titles (plus user keywords)."""
    gamma = "https://gamma-api.polymarket.com"
    clob = "https://clob.polymarket.com"

    session = requests.Session()
    session.headers.update({"User-Agent": "ArbFinder/1.5 (+streamlit)", "Accept": "application/json"})

    # Load a bounded number of markets
    try:
        r = session.get(f"{gamma}/markets", params={"limit": limit_markets, "closed": False}, timeout=25)
        r.raise_for_status()
        js = r.json()
        markets = js if isinstance(js, list) else js.get("data", [])
    except Exception as e:
        st.warning(f"Polymarket Gamma /markets error: {e}")
        return []

    # Prepare candidate keyword sets
    seed_titles = [t for t in candidate_titles if t]
    seed_titles.extend(extra_keywords or [])
    cand_norm = [(t, normalize_title(t), set(normalize_title(t).split())) for t in seed_titles]

    if not cand_norm:
        st.info("Polymarket keyword fetch: no candidate titles/keywords; skipping.")
        if fallback_full_if_empty:
            return fetch_polymarket()
        return []

    # Select matching markets
    selected = []
    for m in markets:
        if not m.get("active", True):
            continue
        if m.get("enableOrderBook") is False:
            continue
        raw = m.get("question") or m.get("title") or m.get("name") or m.get("slug") or ""
        norm = normalize_title(raw)
        tokens = set(norm.split())
        keep = False
        for _, cand_norm_str, cand_tokens in cand_norm:
            if len(tokens.intersection(cand_tokens)) >= min_shared and _sim(norm, cand_norm_str) >= similarity:
                keep = True
                break
        if keep:
            selected.append(m)

    st.caption(f"Polymarket keyword fetch — candidates: {len(cand_norm)}, scanned: {len(markets)}, matched: {len(selected)}")

    if not selected:
        if fallback_full_if_empty:
            st.info("No Polymarket matches via keywords — falling back to full scan.")
            return fetch_polymarket()
        return []

    # Build token ids → price params
    def as_ids(val) -> List[str]:
        if val is None: return []
        if isinstance(val, list): return [str(x) for x in val if x is not None]
        if isinstance(val, dict):
            out = []
            for k in ("YES","yes","Yes","NO","no","No","0","1"):
                if k in val and val[k] is not None: out.append(str(val[k]))
            return out
        if isinstance(val, str):
            try: return as_ids(json.loads(val))
            except Exception: return [s.strip("'\" ") for s in val.strip("[]").split(",") if s.strip()]
        return []

    price_params: List[Dict[str, str]] = []
    token_meta: List[Tuple[str, str, str, str, Optional[str]]] = []

    for m in selected:
        q = m.get("question") or m.get("title") or m.get("name") or m.get("slug") or "(untitled)"
        slug = m.get("slug")
        mid = str(m.get("questionID") or m.get("conditionId") or m.get("id") or slug or "?")
        ids = as_ids(m.get("clobTokenIds") or m.get("tokenIds"))
        if len(ids) != 2:
            outs = m.get("outcomes")
            if isinstance(outs, list) and len(outs) == 2:
                tmp = []
                for o in outs:
                    tid = None
                    if isinstance(o, dict):
                        tid = o.get("clobTokenId") or o.get("tokenId") or o.get("id")
                    tmp.append(str(tid) if tid is not None else "")
                ids = tmp
        if len(ids) != 2 or not ids[0] or not ids[1]:
            continue
        labels = m.get("shortOutcomes")
        if not (isinstance(labels, list) and len(labels) == 2):
            labels = ["Yes", "No"]
        for idx, tid in enumerate(ids[:2]):
            outcome = str(labels[idx]).strip().title() if idx < len(labels) else ("Yes" if idx==0 else "No")
            if outcome not in ("Yes","No"): outcome = "Yes" if idx==0 else "No"
            price_params.append({"token_id": str(tid), "side": "BUY"})
            token_meta.append((mid, q, outcome, str(tid), slug))

    if not price_params:
        return []

    # Fetch prices (chunked, doc shape first)
    def fetch_prices_batch(batch: List[Dict[str,str]]):
        try:
            r = session.post(f"{clob}/prices", json={"params": batch}, timeout=25)
            if r.status_code == 200: return r.json() or {}
            r = session.post(f"{clob}/prices", json=batch, timeout=25)
            if r.status_code == 200: return r.json() or {}
            token_ids = [bp["token_id"] for bp in batch]
            r = session.post(f"{clob}/prices", json={"token_ids": token_ids, "side": "BUY"}, timeout=25)
            if r.status_code == 200: return r.json() or {}
            return {}
        except Exception:
            return {}

    CHUNK = 200
    price_map_all: Dict[str, Any] = {}
    for i in range(0, len(price_params), CHUNK):
        price_map_all.update(fetch_prices_batch(price_params[i:i+CHUNK]))

    quotes: List[Quote] = []
    for (mid, q, outcome, tid, slug) in token_meta:
        entry = price_map_all.get(tid)
        if entry is None: continue
        buy_price = entry.get("BUY") if isinstance(entry, dict) else None
        if buy_price is None and isinstance(entry, (int,float,str)):
            try: buy_price = float(entry)
            except Exception: buy_price = None
        if buy_price is None: continue
        try: p = float(buy_price)
        except Exception: continue
        url_hint = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com/"
        quotes.append(Quote("Polymarket", mid, q, outcome, p, url=url_hint))

    return quotes

@st.cache_data(ttl=60)
def fetch_kalshi(email: str, api_key: str) -> List[Quote]:
    quotes: List[Quote] = []
    if not email or not api_key:
        return quotes
    base = st.session_state.get("kalshi_base", "https://trading-api.kalshi.com/trade-api/v2")
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "kalshi-user": email,
    }
    data = http_get_json(f"{base}/markets", headers=headers)
    if not data:
        return quotes
    markets = data.get("markets") or []
    for m in markets:
        mid = str(m.get("ticker") or m.get("id") or "?")
        name = m.get("title") or m.get("name") or mid
        url = f"https://kalshi.com/markets/{mid}" if mid else None
        orderbook = m.get("orderbook") or {}
        yes_buy = orderbook.get("yes_bid") or orderbook.get("best_yes_offer") or m.get("yes_price")
        no_buy = orderbook.get("no_bid") or orderbook.get("best_no_offer") or m.get("no_price")
        if yes_buy is None and m.get("implied_prob") is not None:
            try: yes_buy = float(m.get("implied_prob"))
            except Exception: pass
        if no_buy is None and yes_buy is not None:
            try: no_buy = 1.0 - float(yes_buy)
            except Exception: pass
        try:
            if yes_buy is not None:
                quotes.append(Quote("Kalshi", mid, name, "Yes", float(yes_buy), url=url))
            if no_buy is not None:
                quotes.append(Quote("Kalshi", mid, name, "No", float(no_buy), url=url))
        except Exception:
            pass
    return quotes

# ==========================
# Matching & Arbitrage
# ==========================

def normalize_title(s: str) -> str:
    s = (s or "").lower()
    aliases = {
        "u.s.": "us", "u.s": "us", "usa": "us", "united states": "us",
        "united kingdom": "uk", "great britain": "uk",
        "gop": "republicans", "dems": "democrats", "pres.": "president",
    }
    for k,v in aliases.items(): s = s.replace(k, v)
    for ch in "?,.:;!@#$%^&*()[]{}|/\\'\"“”’`~+_=": s = s.replace(ch, " ")
    STOP = {"will","the","a","an","of","in","on","to","be","for","by","at","vs","and","or","is",
            "market","contract","question","event","2024","2025"}
    tokens = [t for t in s.split() if t and t not in STOP]
    return " ".join(tokens)

try:
    from rapidfuzz.fuzz import token_set_ratio as rf_ratio
    _USE_RF = True
except Exception:
    _USE_RF = False

def _sim(a: str, b: str) -> float:
    if _USE_RF:
        return rf_ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()

def fuzzy_group_binary(quotes: List[Quote], threshold: float = 0.82, min_shared_keywords: int = 2) -> Dict[str, MatchedMarket]:
    # Build unique markets
    uniq: List[Tuple[str,str,str,str]] = []  # (ex, id, raw, norm)
    seen = set()
    for q in quotes:
        k = (q.exchange, q.market_id)
        if k in seen: continue
        seen.add(k)
        raw = q.market_name or ""
        uniq.append((q.exchange, q.market_id, raw, normalize_title(raw)))

    n = len(uniq)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a,b):
        pa,pb = find(a),find(b)
        if pa!=pb: parent[pb]=pa

    token_sets = [set(t[3].split()) for t in uniq]
    for i in range(n):
        for j in range(i+1,n):
            if uniq[i][0] == uniq[j][0]:
                continue
            if len(token_sets[i].intersection(token_sets[j])) < min_shared_keywords:
                continue
            if _sim(uniq[i][3], uniq[j][3]) >= threshold:
                union(i,j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        p = find(i); clusters.setdefault(p, []).append(i)

    matched: Dict[str, MatchedMarket] = {}
    for idx, members in enumerate(clusters.values()):
        key = f"grp_{idx:04d}"
        members_set = {(uniq[m][0], uniq[m][1]) for m in members}
        legs = [q for q in quotes if (q.exchange, q.market_id) in members_set]
        titles = {}
        for m in members:
            ex, mid, raw, _ = uniq[m]
            titles[ex] = raw
        matched[key] = MatchedMarket(key, titles, legs)
    return matched


def find_surebets(matched: Dict[str, MatchedMarket], fees: Dict[str, FeeModel],
                  cross_exchange_only: bool = True, use_payout_haircut: bool = False) -> List[Arb]:
    arbs: List[Arb] = []
    for grp in matched.values():
        best_yes: Optional[Tuple[Quote, float]] = None
        best_no: Optional[Tuple[Quote, float]] = None
        for q in grp.quotes:
            if q.best_buy_cost is None: continue
            fm = fees.get(q.exchange, FeeModel())
            adj = fm.effective_buy_price(q.best_buy_cost)
            if q.outcome == "Yes":
                if best_yes is None or adj < best_yes[1]: best_yes = (q, adj)
            elif q.outcome == "No":
                if best_no is None or adj < best_no[1]: best_no = (q, adj)
        if not best_yes or not best_no:
            continue
        yes_q, yes_p = best_yes
        no_q, no_p = best_no
        if cross_exchange_only and yes_q.exchange == no_q.exchange:
            continue
        if use_payout_haircut:
            yes_pay = fees.get(yes_q.exchange, FeeModel()).effective_payout(1.0)
            no_pay  = fees.get(no_q.exchange,  FeeModel()).effective_payout(1.0)
            payout = min(yes_pay, no_pay)
        else:
            payout = 1.0
        total_cost = yes_p + no_p
        if total_cost < payout:
            arbs.append(Arb(grp.key, list(grp.titles.values())[0], yes_q, no_q, total_cost, payout - total_cost))
    arbs.sort(key=lambda a: a.margin, reverse=True)
    return arbs

# ==========================
# UI
# ==========================
st.title("🔎 Prediction Market Arbitrage Finder")
st.caption("Polymarket · PredictIt · (Kalshi optional). Binary markets only. Educational use.")

with st.sidebar:
    st.header("Data Sources")
    st.checkbox("Load PredictIt", value=True, key="load_predictit")
    st.checkbox("Load Polymarket (keyword-driven)", value=True, key="load_polymarket_keywords")
    st.checkbox("Fallback: Load Polymarket (full scan)", value=False, key="load_polymarket_full")
    st.checkbox("Load Kalshi (needs credentials)", value=False, key="load_kalshi")

    st.subheader("Polymarket keyword matching")
    pm_sim = st.slider("Similarity threshold (Polymarket title match)", 0.60, 0.95, 0.80, 0.01)
    pm_min_shared = st.slider("Min shared keywords (Polymarket)", 1, 5, 2, 1)
    pm_limit = st.number_input("Gamma markets limit", 50, 2000, 600, 50,
                               help="How many Polymarket markets to scan from Gamma.")
    extra_kw = st.text_input("Extra keywords (comma-separated)", value="",
                             help="Add phrases to search on Polymarket if other sources are empty.")

    st.subheader("Kalshi credentials (optional)")
    email = st.text_input("Email", value="")
    api_key = st.text_input("API Key", value="", type="password")
    st.text_input("Kalshi API base",
                  value="https://trading-api.kalshi.com/trade-api/v2",
                  key="kalshi_base")

    st.subheader("Fuzzy matching (cross-exchange groups)")
    threshold = st.slider("Similarity threshold", 0.60, 0.95, 0.82, 0.01)
    min_shared = st.slider("Min shared keywords", 1, 5, 2, 1)

    st.subheader("Fees (% / $)")
    colA, colB = st.columns(2)
    with colA:
        pi_buy = st.number_input("PredictIt buy load %", 0.0, 20.0, 0.0, 0.1) / 100.0
        pi_out = st.number_input("PredictIt payout haircut %", 0.0, 40.0, 10.0, 0.5) / 100.0
        pi_set = st.number_input("PredictIt settlement $", 0.0, 0.20, 0.00, 0.01)
    with colB:
        pm_buy = st.number_input("Polymarket buy load %", 0.0, 20.0, 0.0, 0.1) / 100.0
        pm_out = st.number_input("Polymarket payout haircut %", 0.0, 40.0, 0.0, 0.5) / 100.0
        pm_set = st.number_input("Polymarket settlement $", 0.0, 0.20, 0.00, 0.01)
    kl_buy = st.number_input("Kalshi buy load %", 0.0, 20.0, 0.0, 0.1) / 100.0
    kl_out = st.number_input("Kalshi payout haircut %", 0.0, 40.0, 0.0, 0.5) / 100.0
    kl_set = st.number_input("Kalshi settlement $", 0.0, 0.20, 0.01, 0.01)

    st.subheader("Budget & Rules")
    budget = st.number_input("Budget ($)", 10.0, 1_000_000.0, 500.0, 10.0)
    require_cross = st.checkbox("Require cross-exchange (Yes & No on different platforms)", value=True)
    apply_haircut = st.checkbox("Apply payout haircut to $1 payoff", value=False)

# Build fee map (make sure this is defined before we use it later)
fees = {
    "PredictIt": FeeModel(buy_load=pi_buy, payout_haircut=pi_out, settlement_fee=pi_set),
    "Polymarket": FeeModel(buy_load=pm_buy, payout_haircut=pm_out, settlement_fee=pm_set),
    "Kalshi":     FeeModel(buy_load=kl_buy, payout_haircut=kl_out, settlement_fee=kl_set),
}

# ==========================
# Load quotes
# ==========================
quotes: List[Quote] = []
progress = st.progress(0, text="Loading markets…")
steps = int(st.session_state.load_predictit) + int(st.session_state.load_polymarket_keywords) + int(st.session_state.load_polymarket_full) + int(st.session_state.load_kalshi)
steps = max(1, steps)
cur = 0

# 1) PredictIt first (provides titles for keyword-driven Polymarket)
if st.session_state.load_predictit:
    cur += 1; progress.progress(cur/steps, text="Loading PredictIt…")
    pi_quotes = fetch_predictit()
    quotes.extend(pi_quotes)
else:
    pi_quotes = []

# 2) Kalshi (optional, also provides titles)
if st.session_state.load_kalshi and email and api_key:
    cur += 1; progress.progress(cur/steps, text="Loading Kalshi…")
    kl_quotes = fetch_kalshi(email, api_key)
    quotes.extend(kl_quotes)
else:
    kl_quotes = []

# Select candidate titles from non-Polymarket sources + user-supplied extras
candidate_titles = sorted({q.market_name for q in (pi_quotes + kl_quotes)})
extra_keywords = [s.strip() for s in extra_kw.split(",") if s.strip()]

# 3) Polymarket via keywords (user-preferred)
if st.session_state.load_polymarket_keywords:
    cur += 1; progress.progress(cur/steps, text="Loading Polymarket (keyword-driven)…")
    pm_quotes_kw = fetch_polymarket_by_keywords(
        candidate_titles,
        extra_keywords,
        similarity=pm_sim,
        min_shared=pm_min_shared,
        limit_markets=pm_limit,
        fallback_full_if_empty=True
    )
    quotes.extend(pm_quotes_kw)

# 4) Optional full-scan fallback
if st.session_state.load_polymarket_full:
    cur += 1; progress.progress(cur/steps, text="Loading Polymarket (full scan)…")
    pm_quotes_full = fetch_polymarket()
    quotes.extend(pm_quotes_full)

progress.progress(1.0, text="Preparing tables…")

if not quotes:
    st.info("No quotes loaded. Check data source toggles or connectivity.")
    st.stop()

# ==========================
# All markets tables
# ==========================
st.subheader("🗂️ All fetched markets & odds")
all_rows = [{
    "Exchange": q.exchange,
    "Market ID": q.market_id,
    "Title": q.market_name,
    "Outcome": q.outcome,
    "Best Buy Cost": q.best_buy_cost,
    "URL": q.url,
} for q in quotes]

df_all = pd.DataFrame(all_rows)
colf1, colf2 = st.columns([2,1])
with colf1:
    query_text = st.text_input("Filter by text (Title contains, case-insensitive)", value="")
with colf2:
    ex_filter = st.multiselect("Exchanges", sorted(df_all["Exchange"].unique()), default=sorted(df_all["Exchange"].unique()))
mask = df_all["Exchange"].isin(ex_filter)
if query_text:
    mask &= df_all["Title"].str.contains(query_text, case=False, na=False)

st.dataframe(df_all[mask].sort_values(["Exchange","Title","Outcome"]).reset_index(drop=True), use_container_width=True)

csv_all = df_all.to_csv(index=False)
st.download_button("Download all markets (CSV)", data=csv_all, file_name="all_markets_odds_flat.csv", mime="text/csv")

try:
    df_pivot = df_all.pivot_table(index=["Exchange","Market ID","Title","URL"], columns="Outcome", values="Best Buy Cost", aggfunc="first").reset_index()
    st.markdown("**Grouped view (one row per market, Yes/No columns):**")
    st.dataframe(df_pivot.sort_values(["Exchange","Title"]).reset_index(drop=True), use_container_width=True)
    csv_grouped = df_pivot.to_csv(index=False)
    st.download_button("Download grouped markets (CSV)", data=csv_grouped, file_name="all_markets_odds_grouped.csv", mime="text/csv")
except Exception:
    pass

# ==========================
# Match & find arbs
# ==========================
matched = fuzzy_group_binary(quotes, threshold=threshold, min_shared_keywords=min_shared)

arbs = find_surebets(matched, fees, cross_exchange_only=require_cross, use_payout_haircut=apply_haircut)
if "arbs" not in locals() or arbs is None:
    arbs = []

st.subheader("📈 Opportunities (risk-free sure-bets)")
if not arbs:
    st.success("No riskless opportunities under current fees/threshold. Tweak similarity, fees, or sources.")
else:
    rows = []
    for a in arbs:
        pairs = a.max_pairs(budget)
        rows.append({
            "Title": a.title_hint,
            "Buy YES @": f"{a.buy_yes.exchange} ({a.buy_yes.best_buy_cost:.3f})",
            "Buy NO @": f"{a.buy_no.exchange} ({a.buy_no.best_buy_cost:.3f})",
            "Adj YES": round(fees[a.buy_yes.exchange].effective_buy_price(a.buy_yes.best_buy_cost), 4),
            "Adj NO": round(fees[a.buy_no.exchange].effective_buy_price(a.buy_no.best_buy_cost), 4),
            "Adj total cost": round(a.total_cost, 4),
            "Adj margin per pair": round(a.per_pair_profit(), 4),
            "Max pairs": pairs,
            "Max profit": round(pairs * a.per_pair_profit(), 2),
            "Links": " | ".join(filter(None, [
                f"YES: {a.buy_yes.url}" if a.buy_yes.url else None,
                f"NO: {a.buy_no.url}" if a.buy_no.url else None,
            ])),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="arbitrage_opportunities.csv", mime="text/csv")

# ==========================
# Possible matches (even if not arbitrage)
# ==========================
st.subheader("🔍 Potential cross-exchange matches (not necessarily arbitrage)")
rows = []
for grp in matched.values():
    exchanges = sorted({q.exchange for q in grp.quotes})
    if len(exchanges) < 2:
        continue  # only care if appears in at least 2 venues
    # Compute best YES/NO per exchange for context
    best_by_ex = {}
    for ex in exchanges:
        yes_prices = [q.best_buy_cost for q in grp.quotes if q.exchange == ex and q.outcome == "Yes" and q.best_buy_cost is not None]
        no_prices  = [q.best_buy_cost for q in grp.quotes if q.exchange == ex and q.outcome == "No"  and q.best_buy_cost is not None]
        best_by_ex[ex] = {
            "YES": min(yes_prices) if yes_prices else None,
            "NO":  min(no_prices)  if no_prices  else None,
        }
    rows.append({
        "Group": grp.key,
        "Titles (by exchange)": grp.titles,
        "Exchanges": ", ".join(exchanges),
        "Best YES/NO per exchange": best_by_ex,
        "Markets": [f"{q.exchange}: {q.market_name}" for q in grp.quotes],
    })
if rows:
    df_matches = pd.DataFrame(rows)
    st.dataframe(df_matches, use_container_width=True)
    csv_matches = df_matches.to_csv(index=False)
    st.download_button("Download possible matches (CSV)", data=csv_matches, file_name="possible_matches.csv", mime="text/csv")
else:
    st.info("No cross-exchange matches found with current similarity/keyword settings.")

# ==========================
# Debug panel
# ==========================
with st.expander("🔧 Review & Debug matching"):
    st.write({k: {
        "titles": v.titles,
        "legs": [ (q.exchange, q.market_id, q.outcome, q.best_buy_cost) for q in v.quotes ]
    } for k, v in matched.items()})
    st.markdown("**Tips**: If mismatches persist, increase similarity, increase min shared keywords, or add aliases in `normalize_title()`." )

# ==========================
# requirements.txt (example)
# ==========================
# streamlit>=1.33
# requests>=2.31
# pandas>=2.0
# numpy>=1.25
# rapidfuzz>=3.9   # optional but recommended for better matching
