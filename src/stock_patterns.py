from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

try:
    import mplfinance as mpf
except ImportError:  # pragma: no cover - handled by user-facing error.
    mpf = None


@dataclass(frozen=True)
class PatternConfig:
    """Detection thresholds used by the classroom rules."""

    max_double_point_gap: float = 0.10


def set_chinese_font() -> None:
    """Set a Windows Chinese font so matplotlib labels do not become boxes."""

    candidates = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
    installed = {font.name for font in font_manager.fontManager.ttflist}

    for font in candidates:
        if font in installed:
            plt.rcParams["font.sans-serif"] = [font]
            break

    plt.rcParams["axes.unicode_minus"] = False


def infer_trend(df: pd.DataFrame) -> pd.Series:
    """Infer TrendForTan from the MA ordering rule on slide 9 of the lecture.

    The CSV does not contain TrendForTan, so classify each trading day:
    - 多頭: MA5 > MA10 > MA20 and Close > MA5.
    - 空頭: MA5 < MA10 < MA20 and Close < MA5.
    - 橫盤整理: all other cases.
    """

    bull = (
        (df["MA5"] > df["MA10"])
        & (df["MA10"] > df["MA20"])
        & (df["Close"] > df["MA5"])
    )
    bear = (
        (df["MA5"] < df["MA10"])
        & (df["MA10"] < df["MA20"])
        & (df["Close"] < df["MA5"])
    )

    trend = pd.Series("橫盤整理", index=df.index, dtype="object")
    trend.loc[bull] = "多頭"
    trend.loc[bear] = "空頭"
    return trend


def load_stock_data_from_csv(
    csv_path: str | Path = "2330_台積電_2025.csv",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Read OHLCV data from CSV and infer Trend when it is not provided."""

    df = pd.read_csv(csv_path)

    if df.empty:
        return df

    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    required = ["Date", "Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA5",
        "MA10",
        "MA20",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if start_date is not None:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df.loc[df.index <= pd.Timestamp(end_date)]

    if "Trend" not in df.columns:
        df["Trend"] = infer_trend(df)

    return df


def load_stock_data(
    stock_code: str = "2330",
    start_date: str = "2025-01-02",
    end_date: str = "2025-12-31",
    csv_path: str | Path = "2330_台積電_2025.csv",
) -> pd.DataFrame:
    """Compatibility wrapper used by the notebook and CLI."""

    _ = stock_code
    return load_stock_data_from_csv(csv_path, start_date, end_date)


def normalize_trend(trend: object) -> str:
    """Convert database trend labels into up/down/flat labels."""

    if trend in {"多頭", "上漲", "上漲趨勢"}:
        return "up"
    if trend in {"空頭", "下跌", "下跌趨勢"}:
        return "down"
    return "flat"


def get_turning_points(df: pd.DataFrame) -> list[dict]:
    """Find one representative turning point from each continuous trend group.

    上漲: take the highest high as a peak.
    下跌: take the lowest low as a valley.
    橫盤: take the close nearest to the group median as a flat point.
    """

    if df.empty:
        return []

    work = df.copy()
    work["TrendNorm"] = work["Trend"].apply(normalize_trend)
    work["TrendGroup"] = (work["TrendNorm"] != work["TrendNorm"].shift()).cumsum()

    points: list[dict] = []

    for group_id, group in work.groupby("TrendGroup"):
        trend = group["TrendNorm"].iloc[0]

        if trend == "up":
            row = group.loc[group["High"].idxmax()]
            points.append(
                {
                    "type": "peak",
                    "date": row.name,
                    "price": float(row["High"]),
                    "group": int(group_id),
                }
            )
        elif trend == "down":
            row = group.loc[group["Low"].idxmin()]
            points.append(
                {
                    "type": "valley",
                    "date": row.name,
                    "price": float(row["Low"]),
                    "group": int(group_id),
                }
            )
        else:
            median_close = group["Close"].median()
            nearest_index = (group["Close"] - median_close).abs().idxmin()
            row = group.loc[nearest_index]
            points.append(
                {
                    "type": "flat",
                    "date": row.name,
                    "price": float(row["Close"]),
                    "group": int(group_id),
                }
            )

    return points


def clean_zigzag(points: Iterable[dict]) -> list[dict]:
    """Build the ZigZag points using the lecture segment algorithm.

    Lecture slides 14-18 do not only merge adjacent equal labels. They extend a
    segment while prices keep moving in the same direction from the segment
    start. When the price direction reverses, the segment start and previous
    point become valid ZigZag endpoints.
    """

    ordered_points = list(points)
    if len(ordered_points) <= 2:
        return ordered_points

    segments: list[tuple[dict, dict]] = []
    segment = [ordered_points[0]]

    for i in range(1, len(ordered_points)):
        curr = ordered_points[i]
        prev = ordered_points[i - 1]
        start = segment[0]

        same_direction = (curr["price"] - prev["price"]) * (
            prev["price"] - start["price"]
        ) >= 0

        if same_direction:
            segment.append(curr)
        else:
            if len(segment) >= 2:
                segments.append((segment[0], segment[-1]))
            segment = [prev, curr]

    if len(segment) >= 2:
        segments.append((segment[0], segment[-1]))

    # Ordered dict behavior: keep the first occurrence for each date while
    # preserving the order generated by the segment endpoints.
    zigzag_by_date: dict[pd.Timestamp, dict] = {}
    for start, end in segments:
        zigzag_by_date.setdefault(pd.Timestamp(start["date"]), start)
        zigzag_by_date.setdefault(pd.Timestamp(end["date"]), end)

    return list(zigzag_by_date.values())


def detect_w_m_patterns(
    points: list[dict],
    df: pd.DataFrame | None = None,
    config: PatternConfig | None = None,
    require_confirmation: bool = True,
) -> list[dict]:
    """Detect W-bottom and M-top patterns from every 5-point window.

    This follows the lecture notes:
    - W Bottom: a/c/e are peak, b/d are valley or flat, b-d gap <= 10%, a >= c.
      A later close above c is recorded as confirmation.
    - M Top: a/c/e are valley, b/d are peak or flat, b-d gap <= 10%, a <= c.
      A later close below c is recorded as confirmation.

    The default requires confirmation because the lecture says the pattern is
    established only after neckline break / breakdown.
    """

    config = config or PatternConfig()
    patterns: list[dict] = []

    for i in range(len(points) - 4):
        p1, p2, p3, p4, p5 = points[i : i + 5]

        is_w_bottom = (
            p1["type"] == "peak"
            and p2["type"] in ["valley", "flat"]
            and p3["type"] == "peak"
            and p4["type"] in ["valley", "flat"]
            and p5["type"] == "peak"
            and abs(p2["price"] - p4["price"]) / max(abs(p2["price"]), 1)
            <= config.max_double_point_gap
            and p1["price"] >= p3["price"]
        )

        if is_w_bottom:
            confirm_date = p5["date"]
            confirm_price = p5["price"]
            confirmed = p5["price"] > p3["price"]

            if df is not None:
                future = df.loc[df.index >= p5["date"]]
                break_rows = future[future["Close"] > p3["price"]]
                if break_rows.empty:
                    if require_confirmation:
                        continue
                else:
                    confirm_date = break_rows.index[0]
                    confirm_price = float(break_rows.iloc[0]["Close"])
                    confirmed = True
            elif require_confirmation and not confirmed:
                continue

            patterns.append(
                {
                    "pattern": "W底",
                    "points": [p1, p2, p3, p4, p5],
                    "start": p1["date"],
                    "end": p5["date"],
                    "neckline": p3["price"],
                    "confirm_date": confirm_date,
                    "confirm_price": confirm_price,
                    "confirmed": confirmed,
                }
            )

        is_m_top = (
            p1["type"] == "valley"
            and p2["type"] in ["peak", "flat"]
            and p3["type"] == "valley"
            and p4["type"] in ["peak", "flat"]
            and p5["type"] == "valley"
            and abs(p2["price"] - p4["price"]) / max(abs(p2["price"]), 1)
            <= config.max_double_point_gap
            and p1["price"] <= p3["price"]
        )

        if is_m_top:
            confirm_date = p5["date"]
            confirm_price = p5["price"]
            confirmed = p5["price"] < p3["price"]

            if df is not None:
                future = df.loc[df.index >= p5["date"]]
                break_rows = future[future["Close"] < p3["price"]]
                if break_rows.empty:
                    if require_confirmation:
                        continue
                else:
                    confirm_date = break_rows.index[0]
                    confirm_price = float(break_rows.iloc[0]["Close"])
                    confirmed = True
            elif require_confirmation and not confirmed:
                continue

            patterns.append(
                {
                    "pattern": "M頭",
                    "points": [p1, p2, p3, p4, p5],
                    "start": p1["date"],
                    "end": p5["date"],
                    "neckline": p3["price"],
                    "confirm_date": confirm_date,
                    "confirm_price": confirm_price,
                    "confirmed": confirmed,
                }
            )

    return patterns


def neckline_value(p_start: dict, p_end: dict, target_date: pd.Timestamp) -> float:
    """Project a sloped neckline from two turning points to target_date."""

    days_total = (p_end["date"] - p_start["date"]).days
    if days_total == 0:
        return float(p_start["price"])

    slope = (p_end["price"] - p_start["price"]) / days_total
    days_target = (target_date - p_start["date"]).days
    return float(p_start["price"] + slope * days_target)


def detect_triple_patterns(
    points: list[dict],
    df: pd.DataFrame | None = None,
    config: PatternConfig | None = None,
) -> list[dict]:
    """Detect triple-top and triple-bottom patterns from every 7-point window.

    This follows the lecture notes exactly:
    - Triple Top: p1/p3/p5/p7 are valley; p2/p4/p6 are peak or flat.
      The neckline connects p3 and p5, and p7 close must be below that line.
    - Triple Bottom: p1/p3/p5/p7 are peak; p2/p4/p6 are valley or flat.
      The neckline connects p3 and p5, and p7 close must be above that line.
    """

    config = config or PatternConfig()
    patterns: list[dict] = []

    for i in range(len(points) - 6):
        p1, p2, p3, p4, p5, p6, p7 = points[i : i + 7]

        is_triple_top_shape = (
            p1["type"] == "valley"
            and p3["type"] == "valley"
            and p5["type"] == "valley"
            and p7["type"] == "valley"
            and p2["type"] in ["peak", "flat"]
            and p4["type"] in ["peak", "flat"]
            and p6["type"] in ["peak", "flat"]
        )

        if is_triple_top_shape:
            neckline_at_p7 = neckline_value(p3, p5, p7["date"])
            confirm_price = p7["price"]
            if df is not None and p7["date"] in df.index:
                confirm_price = float(df.loc[p7["date"], "Close"])

            if confirm_price < neckline_at_p7:
                patterns.append(
                    {
                        "pattern": "三重頂",
                        "points": [p1, p2, p3, p4, p5, p6, p7],
                        "start": p1["date"],
                        "end": p7["date"],
                        "neckline_points": [p3, p5],
                        "neckline_at_confirm": neckline_at_p7,
                        "confirm_date": p7["date"],
                        "confirm_price": confirm_price,
                    }
                )

        is_triple_bottom_shape = (
            p1["type"] == "peak"
            and p3["type"] == "peak"
            and p5["type"] == "peak"
            and p7["type"] == "peak"
            and p2["type"] in ["valley", "flat"]
            and p4["type"] in ["valley", "flat"]
            and p6["type"] in ["valley", "flat"]
        )

        if is_triple_bottom_shape:
            neckline_at_p7 = neckline_value(p3, p5, p7["date"])
            confirm_price = p7["price"]
            if df is not None and p7["date"] in df.index:
                confirm_price = float(df.loc[p7["date"], "Close"])

            if confirm_price > neckline_at_p7:
                patterns.append(
                    {
                        "pattern": "三重底",
                        "points": [p1, p2, p3, p4, p5, p6, p7],
                        "start": p1["date"],
                        "end": p7["date"],
                        "neckline_points": [p3, p5],
                        "neckline_at_confirm": neckline_at_p7,
                        "confirm_date": p7["date"],
                        "confirm_price": confirm_price,
                    }
                )

    return patterns


def summarize_patterns(patterns: list[dict]) -> pd.DataFrame:
    """Create a compact table for notebook display and README screenshots."""

    rows = []
    for pattern in patterns:
        rows.append(
            {
                "pattern": pattern["pattern"],
                "start": pattern["start"].date(),
                "end": pattern["end"].date(),
                "confirm_date": pattern["confirm_date"].date(),
                "confirm_price": round(float(pattern["confirm_price"]), 2),
                "confirmed": pattern.get("confirmed", True),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pattern",
            "start",
            "end",
            "confirm_date",
            "confirm_price",
            "confirmed",
        ],
    )


def select_overview_patterns(patterns: list[dict]) -> list[dict]:
    """Keep the main non-overlapping patterns for lecture-style overview plots.

    The lecture code scans every window, but the overview slide visually presents
    the strongest/main structure instead of drawing every overlapping candidate.
    This keeps the longest confirmed candidate per pattern type.
    """

    selected: list[dict] = []
    for pattern_name in sorted({pattern["pattern"] for pattern in patterns}):
        candidates = [p for p in patterns if p["pattern"] == pattern_name]
        if not candidates:
            continue
        candidates.sort(
            key=lambda p: (
                (p["end"] - p["start"]).days,
                p["confirm_price"],
            ),
            reverse=True,
        )
        selected.append(candidates[0])
    return sorted(selected, key=lambda p: p["start"])


def summarize_windows(points: list[dict], window_size: int) -> pd.DataFrame:
    """Show every ZigZag window for checking against the lecture rules."""

    rows = []
    labels = list("abcdefg")[:window_size]

    for start_index in range(len(points) - window_size + 1):
        window = points[start_index : start_index + window_size]
        row = {"window": start_index + 1}
        for label, point in zip(labels, window):
            row[f"{label}_type"] = point["type"]
            row[f"{label}_date"] = point["date"].date()
            row[f"{label}_price"] = round(float(point["price"]), 2)
        rows.append(row)

    return pd.DataFrame(rows)


def _series_from_points(index: pd.DatetimeIndex, points: list[dict]) -> pd.Series:
    """Build a connected line series on mplfinance candle coordinates.

    mplfinance does not connect points across NaN gaps. To make the line look
    like the lecture slides, each pair of turning points is interpolated across
    the trading-day rows between them.
    """

    series = pd.Series(index=index, dtype="float64")

    valid_points = [
        (pd.Timestamp(point["date"]), float(point["price"]))
        for point in points
        if pd.Timestamp(point["date"]) in series.index
    ]

    if len(valid_points) == 1:
        series.loc[valid_points[0][0]] = valid_points[0][1]
        return series

    for (start_date, start_price), (end_date, end_price) in zip(
        valid_points,
        valid_points[1:],
    ):
        start_pos = series.index.get_loc(start_date)
        end_pos = series.index.get_loc(end_date)

        if end_pos < start_pos:
            start_pos, end_pos = end_pos, start_pos
            start_price, end_price = end_price, start_price

        span = end_pos - start_pos
        if span == 0:
            series.iloc[start_pos] = start_price
            continue

        for offset, row_pos in enumerate(range(start_pos, end_pos + 1)):
            ratio = offset / span
            series.iloc[row_pos] = start_price + (end_price - start_price) * ratio

    return series


def _horizontal_series(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    value: float,
) -> pd.Series:
    """Build a horizontal neckline series between start and end."""

    series = pd.Series(index=index, dtype="float64")
    mask = (series.index >= start) & (series.index <= end)
    series.loc[mask] = float(value)
    return series


def _sloped_neckline_series(
    index: pd.DatetimeIndex,
    p_start: dict,
    p_end: dict,
    confirm_date: pd.Timestamp,
) -> pd.Series:
    """Build a sloped neckline series projected to the confirmation date."""

    series = pd.Series(index=index, dtype="float64")
    start = pd.Timestamp(p_start["date"])
    end = pd.Timestamp(confirm_date)
    mask = (series.index >= start) & (series.index <= end)

    for date in series.index[mask]:
        series.loc[date] = neckline_value(p_start, p_end, date)

    return series


def _pattern_colors(pattern_name: str) -> tuple[str, str]:
    """Return ZigZag and neckline colors similar to the lecture slides."""

    if pattern_name == "W底":
        return "darkgreen", "green"
    if pattern_name == "M頭":
        return "darkred", "red"
    if pattern_name == "三重底":
        return "blue", "steelblue"
    if pattern_name == "三重頂":
        return "orange", "darkorange"
    return "purple", "gray"


def pattern_label(pattern_name: str) -> str:
    """Return English labels for chart output."""

    labels = {
        "W底": "W Bottom",
        "M頭": "M Top",
        "三重底": "Triple Bottom",
        "三重頂": "Triple Top",
    }
    return labels.get(pattern_name, pattern_name)


def plot_pattern(df: pd.DataFrame, pattern: dict, stock_code: str = "2330") -> None:
    """Plot one detected pattern using candlesticks, ZigZag, and neckline."""

    if mpf is None:
        raise ImportError("Please install mplfinance: pip install mplfinance")

    plot_df = df.loc[pattern["start"] : pattern["end"]].copy()
    zigzag_color, neckline_color = _pattern_colors(pattern["pattern"])
    zigzag_series = _series_from_points(plot_df.index, pattern["points"])

    addplots = [
        mpf.make_addplot(
            zigzag_series,
            type="line",
            color=zigzag_color,
            width=2.0,
            marker="o",
            markersize=5,
        )
    ]

    if pattern["pattern"] in ["W底", "M頭"]:
        neckline_series = _horizontal_series(
            plot_df.index,
            pattern["start"],
            pattern["end"],
            pattern["neckline"],
        )
    else:
        n1, n2 = pattern["neckline_points"]
        neckline_series = _sloped_neckline_series(
            plot_df.index,
            n1,
            n2,
            pattern["confirm_date"],
        )

    addplots.append(
        mpf.make_addplot(
            neckline_series,
            type="line",
            color=neckline_color,
            linestyle="--",
            width=1.4,
        )
    )

    mpf.plot(
        plot_df,
        type="candle",
        style="yahoo",
        volume=False,
        addplot=addplots,
        figsize=(12, 5.5),
        title=f"{stock_code} {pattern_label(pattern['pattern'])} {pattern['start'].date()} -> {pattern['end'].date()}",
        ylabel="Price",
        warn_too_much_data=10000,
    )


def plot_overview(
    df: pd.DataFrame,
    patterns: list[dict],
    stock_code: str = "2330",
    title: str | None = None,
) -> None:
    """Plot a candlestick overview with all detected pattern overlays."""

    if mpf is None:
        raise ImportError("Please install mplfinance: pip install mplfinance")

    addplots = []
    for pattern in patterns:
        zigzag_color, neckline_color = _pattern_colors(pattern["pattern"])
        addplots.append(
            mpf.make_addplot(
                _series_from_points(df.index, pattern["points"]),
                type="line",
                color=zigzag_color,
                width=2.0,
                marker="o",
                markersize=4,
            )
        )
        if pattern["pattern"] in ["W底", "M頭"]:
            neckline_series = _horizontal_series(
                df.index,
                pattern["start"],
                pattern["end"],
                pattern["neckline"],
            )
        else:
            n1, n2 = pattern["neckline_points"]
            neckline_series = _sloped_neckline_series(
                df.index,
                n1,
                n2,
                pattern["confirm_date"],
            )
        addplots.append(
            mpf.make_addplot(
                neckline_series,
                type="line",
                color=neckline_color,
                linestyle="--",
                width=1.2,
            )
        )

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        volume=False,
        addplot=addplots,
        figsize=(16, 7),
        title=title or f"{stock_code} Pattern Overview",
        ylabel="Price",
        warn_too_much_data=10000,
    )
