"""Strict parsing of product purity lower bounds, shared by chat and planning."""
from decimal import Decimal, InvalidOperation
import re
import unicodedata


def parse_purity_target(value):
    """Accept one fraction/percentage, optionally phrased as a lower bound.

    Full matching prevents ranges, upper bounds and multiple numbers from being
    silently converted into a different optimization goal.
    """
    if isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).strip()
        match = re.fullmatch(
            r"(?:纯度\s*(?:为|是)?\s*)?"
            r"(?:(?:不低于|至少|达到|>=|≥)\s*)?"
            r"(?P<number>[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"\s*(?P<percent>%)?\s*(?:(?:及)?以上)?", text)
        if not match:
            raise ValueError("纯度目标格式无效，请使用 99.99%、0.9999 或 0.9999以上；当前支持纯度下限。")
        number, percent = match.group("number"), bool(match.group("percent"))
    elif type(value) in (int, float):
        number, percent = str(value), False
    else:
        raise ValueError("纯度目标格式无效，请使用 99.99% 或 0.9999。")
    try:
        fraction = Decimal(number)
        if percent:
            fraction /= Decimal(100)
        if not fraction.is_finite() or not 0 <= fraction <= 1:
            raise ValueError("纯度目标必须在 0 到 1 之间，百分数请保留 %。")
    except InvalidOperation:
        raise ValueError("纯度目标格式无效，请使用 99.99% 或 0.9999。") from None
    return float(fraction)
