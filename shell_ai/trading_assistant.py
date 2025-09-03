import logging
from typing import Dict

import yfinance as yf

logger = logging.getLogger(__name__)

def evaluate(symbol: str, portfolio: Dict[str, float]) -> Dict[str, float]:
    """Evaluate a simple trading strategy for the given symbol.

    A very naive strategy is implemented:
    - Buy one share when the closing price is below 100.
    - Sell one share when the closing price is above 100.

    Parameters
    ----------
    symbol: str
        The ticker symbol to evaluate.
    portfolio: dict
        A mapping with ``cash`` and ``shares`` to be updated in place.

    Returns
    -------
    dict
        The updated portfolio.
    """
    data = yf.Ticker(symbol).history(period="1mo")

    for price in data["Close"]:
        if price < 100 and portfolio.get("cash", 0) >= price:
            portfolio["cash"] -= price
            portfolio["shares"] = portfolio.get("shares", 0) + 1
            logger.info(f"Bought 1 share of {symbol} at {price}")
        elif price > 100 and portfolio.get("shares", 0) > 0:
            portfolio["cash"] += price
            portfolio["shares"] -= 1
            logger.info(f"Sold 1 share of {symbol} at {price}")
    return portfolio
