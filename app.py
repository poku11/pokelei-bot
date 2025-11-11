# vinted_app.py
"""
Vinted Buy/Resell Scanner (Streamlit)
- Collector asynchrone (httpx + lxml)
- Moteur de scoring (profit, velocity, risk, score final)
- Interface Streamlit pour scanner une URL Vinted et afficher TOP N résultats
Usage:
  pip install -r requirements.txt
  streamlit run vinted_app.py
Notes:
 - Adapte les XPATH/CSS selectors dans CollectorAsync._parse_search_html_lxml si nécessaire
 - Respecte les ToS de Vinted (rate limit, pas d'automatisation d'achat)
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import math, re, asyncio, time
import httpx
from lxml import html as lxml_html

import streamlit as st
import pandas as pd

# -----------------------
# Config
# -----------------------
USER_AGENT = "Mozilla/5.0 (compatible; VintedBot/1.0; +https://example.com/bot)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"}

DEFAULT_CONCURRENCY = 6
DEFAULT_TIMEOUT = 15.0

VINTED_FEES_PCT = 0.10
AVG_SHIPPING_COST = 5.0
DEFAULT_REFURB_COST = 3.0

MIN_PROFIT_DESIRABLE = 5.0
MAX_PROFIT_CONSIDERED = 200.0

WEIGHT_PROFIT = 0.40
WEIGHT_VELOCITY = 0.55
WEIGHT_RISK = 0.15

# -----------------------
# Utils / scoring engine
# -----------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def median(nums: List[float]) -> Optional[float]:
    if not nums:
        return None
    s = sorted(nums)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n//2 - 1] + s[n//2]) / 2.0)

def estimate_market_price(historical_prices: List[float]) -> Optional[float]:
    return median(historical_prices)

def compute_net_profit(asking_price: float,
                       market_price_est: Optional[float],
                       fees_pct: float = VINTED_FEES_PCT,
                       shipping: float = AVG_SHIPPING_COST,
                       refurb: float = DEFAULT_REFURB_COST) -> Optional[float]:
    if market_price_est is None:
        return None
    gross = market_price_est - asking_price
    fees = market_price_est * fees_pct
    net = gross - fees - shipping - refurb
    return round(net, 2)

def velocity_score(asking_price: float,
                   market_price_est: Optional[float],
                   posted_at: Optional[str] = None,
                   likes: int = 0,
                   views: int = 0,
                   brand_popularity: float = 0.5) -> float:
    if market_price_est is None:
        return 0.0
    price_ratio = asking_price / market_price_est if market_price_est > 0 else 1.0
    price_factor = clamp(1.5 - price_ratio, 0.0, 1.5) / 1.5

    recency_factor = 0.0
    if posted_at:
        try:
            dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
            recency_factor = clamp(1 - (delta_hours / (24 * 7)), 0.0, 1.0)
        except Exception:
            recency_factor = 0.0

    social = math.log(1 + likes + views / 20.0) / 5.0
    social = clamp(social, 0.0, 1.0)
    brand = clamp(brand_popularity, 0.0, 1.0)

    score = 0.5 * price_factor + 0.3 * recency_factor + 0.15 * social + 0.05 * brand
    return clamp(score, 0.0, 1.0)

def risk_penalty(photo_quality: float = 1.0,
                 ambiguous_brand: bool = False,
                 suspect_low_price: bool = False) -> float:
    r = 0.0
    if photo_quality < 0.5:
        r += 0.25
    if ambiguous_brand:
        r += 0.35
    if suspect_low_price:
        r += 0.25
    return clamp(r, 0.0, 1.0)

def final_score(net_profit: Optional[float],
                velocity: float,
                risk: float,
                min_profit: float = MIN_PROFIT_DESIRABLE,
                max_profit: float = MAX_PROFIT_CONSIDERED) -> Dict[str, Any]:
    if net_profit is None:
        profit_norm = 0.0
    else:
        profit_norm = (net_profit - min_profit) / (max_profit - min_profit)
        profit_norm = clamp(profit_norm, 0.0, 1.0)

    raw = WEIGHT_PROFIT * profit_norm + WEIGHT_VELOCITY * velocity - WEIGHT_RISK * risk
    raw = clamp(raw, 0.0, 1.0)
    score_pct = round(raw * 100, 2)

    return {
        "score": score_pct,
        "profit_normalized": round(profit_norm, 4),
        "velocity": round(velocity, 4),
        "risk": round(risk, 4),
        "net_profit": None if net_profit is None else round(net_profit, 2)
    }

def score_listing(listing: Dict[str, Any]) -> Dict[str, Any]:
    asking = float(listing.get("asking_price", 0.0))
    market_est = listing.get("market_price_est")
    if market_est is None:
        market_est = estimate_market_price(listing.get("historical_prices", []))
    netp = compute_net_profit(asking, market_est)
    vel = velocity_score(asking, market_est,
                         posted_at=listing.get("posted_at"),
                         likes=int(listing.get("likes", 0)),
                         views=int(listing.get("views", 0)),
                         brand_popularity=float(listing.get("brand_popularity", 0.5)))
    risk = risk_penalty(photo_quality=float(listing.get("photo_quality", 1.0)),
                        ambiguous_brand=bool(listing.get("ambiguous_brand", False)),
                        suspect_low_price=bool(listing.get("suspect_low_price", False)))

    final = final_score(netp, vel, risk)
    return {
        "asking_price": asking,
        "market_price_est": market_est,
        "net_profit": final["net_profit"],
        "score": final["score"],
        "components": {
            "profit_normalized": final["profit_normalized"],
            "velocity": final["velocity"],
            "risk": final["risk"]
        },
        "raw": {
            "velocity_raw": vel,
            "risk_raw": risk,
            "net_profit_raw": netp
        }
    }

# -----------------------
# Async Collector (httpx + lxml)
# -----------------------
class CollectorAsync:
    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY, headers: dict = None, timeout: float = DEFAULT_TIMEOUT):
        self.concurrency = concurrency
        self.sema = asyncio.Semaphore(concurrency)
        self.headers = headers or DEFAULT_HEADERS
        self.timeout = timeout
        self._client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)

    async def close(self):
        await self._client.aclose()

    async def _fetch(self, url: str) -> Optional[str]:
        async with self.sema:
            try:
                r = await self._client.get(url)
                if r.status_code == 200:
                    return r.text
                else:
