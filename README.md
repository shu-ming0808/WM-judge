# 0525 Stock Pattern Detection Homework

This project implements the 0525 classroom exercises. Python reads `2330_台積電_2025.csv`, detects patterns, and renders charts in a notebook.

- Exercise 1: W Bottom / M Top
- Exercise 2: Triple Bottom / Triple Top

The main presentation is in Jupyter Notebook. Reusable functions are stored in `src/stock_patterns.py` so the project is ready to push to GitHub.

## Environment

- Python 3.10+
- CSV source: `2330_台積電_2025.csv`
- Required CSV columns: `Date/date`, `Open`, `High`, `Low`, `Close`, `Volume`, `MA5`, `MA10`, `MA20`
- If the CSV does not contain `Trend`, the project follows slide 9 of the 0518 lecture: `MA5 > MA10 > MA20` and `Close > MA5` means a bull trend; the reversed inequalities mean a bear trend; every other case means a sideways trend.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name db-0525-homework --display-name "DB 0525 Homework"
```

## Notebook

用 VSCode 開啟：

```text
notebooks/0525_patterns.ipynb
```

The notebook will:

1. Read daily OHLCV data from `2330_台積電_2025.csv`.
2. Infer `Trend` from the 0518 lecture's MA-ordering rule when needed.
3. Build turning points from the `Trend` column.
4. Clean the ZigZag sequence.
5. Detect W Bottom / M Top patterns.
6. Detect Triple Bottom / Triple Top patterns.
7. Render individual pattern charts and an annual overview chart.

## Command Line

也可以直接跑：

```powershell
python pattern_detection_0525.py
```

## Project Structure

```text
.
├── notebooks/
│   └── 0525_patterns.ipynb
├── src/
│   ├── __init__.py
│   └── stock_patterns.py
├── pattern_detection_0525.py
├── requirements.txt
├── VERSION
├── .gitignore
└── README.md
```

## Git

Initialize and commit:

```powershell
git init
git add .
git commit -m "Add 0525 stock pattern homework"
```

If the GitHub repository already exists:

```powershell
git remote add origin <你的 GitHub repo URL>
git branch -M main
git push -u origin main
```
