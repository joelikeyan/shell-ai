"""Top-level package for shell-ai."""

__all__ = ["TradingAssistant"]

def __getattr__(name: str):
    if name == "TradingAssistant":
        from .trading_assistant import TradingAssistant
        return TradingAssistant
    raise AttributeError(f"module {__name__} has no attribute {name!r}")
