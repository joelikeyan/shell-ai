"""Autonomous stock trading assistant.

This module provides a ``TradingAssistant`` class that can analyse
stocks, generate predictions and manage a simple portfolio.  It is a
basic, example implementation and is not intended for production use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression


@dataclass
class Position:
    """Represents a position in the portfolio."""

    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    sector: str


@dataclass
class Portfolio:
    """Simple portfolio with basic risk management."""

    capital: float
    positions: Dict[str, Position] = field(default_factory=dict)
    sector_allocation: Dict[str, float] = field(default_factory=dict)

    def _can_allocate(self, cost: float, sector: str, max_sector_alloc: float = 0.5) -> bool:
        """Return True when the cost fits within cash and sector limits."""

        sector_total = self.sector_allocation.get(sector, 0.0)
        total_equity = self.capital + sum(self.sector_allocation.values())
        sector_limit = total_equity * max_sector_alloc
        return cost <= self.capital and sector_total + cost <= sector_limit

    def enter_position(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        target: float,
        sector: str,
        max_risk: float = 0.02,
    ) -> Position | None:
        risk_per_share = price - stop_loss
        if risk_per_share <= 0:
            return None
        max_loss = self.capital * max_risk
        quantity = int(max_loss / risk_per_share)
        cost = quantity * price
        if quantity <= 0 or not self._can_allocate(cost, sector):
            return None
        self.capital -= cost
        self.positions[symbol] = Position(symbol, quantity, price, stop_loss, target, sector)
        self.sector_allocation[sector] = self.sector_allocation.get(sector, 0.0) + cost
        return self.positions[symbol]

    def exit_position(self, symbol: str, price: float) -> float | None:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None
        proceeds = pos.quantity * price
        self.capital += proceeds
        self.sector_allocation[pos.sector] -= pos.quantity * pos.entry_price
        if self.sector_allocation[pos.sector] <= 0:
            del self.sector_allocation[pos.sector]
        return proceeds - pos.quantity * pos.entry_price

    def dashboard(self) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for pos in self.positions.values():
            current_price = yf.Ticker(pos.symbol).history(period="1d")['Close'][0]
            rows.append(
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                    "unrealized": (current_price - pos.entry_price) * pos.quantity,
                }
            )
        return pd.DataFrame(rows)


class TradingAssistant:
    """Autonomous trading assistant with prediction and execution helpers."""

    def __init__(self, capital: float = 100_000) -> None:
        self.portfolio = Portfolio(capital)

    @staticmethod
    def _download(symbol: str) -> pd.DataFrame:
        return yf.download(symbol, period="1y", progress=False)

    def scan_market(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        return {sym: self._download(sym) for sym in symbols}

    @staticmethod
    def _linear_regression_prediction(prices: pd.Series, horizon: int) -> Tuple[float, float]:
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        model = LinearRegression().fit(x, y)
        future_x = np.arange(len(prices), len(prices) + horizon).reshape(-1, 1)
        pred = model.predict(future_x)[-1]
        confidence = model.score(x, y)
        return float(pred), float(confidence)

    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
        closes = data['Close']
        preds: Dict[int, Tuple[float, float]] = {}
        for horizon in (30, 60, 90):
            preds[horizon] = self._linear_regression_prediction(closes, horizon)
        return preds

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['VolumeAvg'] = df['Volume'].rolling(20).mean()
        return df

    def generate_signal(self, data: pd.DataFrame) -> str | None:
        row = data.iloc[-1]
        if row['SMA50'] > row['SMA200'] and row['RSI'] < 70 and row['Close'] > row['SMA50']:
            return 'buy'
        if row['SMA50'] < row['SMA200'] or row['RSI'] > 70:
            return 'sell'
        return None

    @staticmethod
    def _sector(symbol: str) -> str:
        info = yf.Ticker(symbol).info
        return info.get('sector', 'Unknown')

    def evaluate(self, symbol: str) -> None:
        data = self._add_indicators(self._download(symbol))
        preds = self.predict(symbol, data)
        signal = self.generate_signal(data)
        price = float(data['Close'].iloc[-1])
        sector = self._sector(symbol)
        stop_loss = price * 0.95
        target = price * 1.10
        if signal == 'buy':
            pos = self.portfolio.enter_position(symbol, price, stop_loss, target, sector)
            if pos:
                direction = 'upward' if preds[30][0] > price else 'downward'
                confidence = int(preds[30][1] * 100)
                print(
                    f"Predicted 30-day {direction} trend in {symbol} ({confidence}% confidence). "
                    f"Entering position at ${price:.2f} with stop-loss at ${stop_loss:.2f} and target at ${target:.2f}."
                )
        elif signal == 'sell' and symbol in self.portfolio.positions:
            pnl = self.portfolio.exit_position(symbol, price)
            print(f"Exit signal for {symbol}. Closing position at ${price:.2f}. PnL: ${pnl:.2f}")
        else:
            print(f"No action for {symbol}.")

    def portfolio_dashboard(self) -> pd.DataFrame:
        return self.portfolio.dashboard()

    def daily_summary(self) -> None:
        dashboard = self.portfolio_dashboard()
        if dashboard.empty:
            print("No positions held.")
            return
        print(dashboard)
        print(f"Remaining capital: ${self.portfolio.capital:.2f}")
