from src.stock_patterns import (
    clean_zigzag,
    detect_triple_patterns,
    detect_w_m_patterns,
    get_turning_points,
    load_stock_data,
    plot_overview,
    plot_pattern,
    select_overview_patterns,
    set_chinese_font,
    summarize_patterns,
)


STOCK_CODE = "2330"
START_DATE = "2025-01-02"
END_DATE = "2025-12-31"


def main() -> None:
    """Run the full 0525 homework workflow from the command line."""

    set_chinese_font()

    df = load_stock_data(
        stock_code=STOCK_CODE,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    if df.empty:
        print("查無資料，請確認股票代號與日期區間。")
        return

    points = get_turning_points(df)
    zigzag_points = clean_zigzag(points)
    wm_patterns = detect_w_m_patterns(zigzag_points, df=df)
    triple_patterns = detect_triple_patterns(zigzag_points, df=df)
    all_patterns = select_overview_patterns(wm_patterns) + select_overview_patterns(
        triple_patterns
    )

    print(f"股票代號：{STOCK_CODE}")
    print(f"資料期間：{START_DATE} ~ {END_DATE}")
    print(f"資料筆數：{len(df)}")
    print(f"原始轉折點數：{len(points)}")
    print(f"整理後 ZigZag 點數：{len(zigzag_points)}")

    print("\n=== W底 / M頭 ===")
    print(summarize_patterns(wm_patterns) if wm_patterns else "沒有偵測到型態")

    print("\n=== 三重頂 / 三重底 ===")
    print(summarize_patterns(triple_patterns) if triple_patterns else "沒有偵測到型態")

    for pattern in all_patterns:
        plot_pattern(df, pattern, stock_code=STOCK_CODE)

    if all_patterns:
        plot_overview(df, all_patterns, stock_code=STOCK_CODE)


if __name__ == "__main__":
    main()
