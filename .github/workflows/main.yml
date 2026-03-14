name: Turuhashi AI Strategy Report

on:
  schedule:
    - cron: '0 20 * * *'
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11' # 3.12より安定している3.11を推奨
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          # tvdatafeedをURLから直接、かつ強制的にインストール
          pip install git+https://github.com/StreamAlpha/tvdatafeed.git@master || pip install git+https://github.com/StreamAlpha/tvdatafeed.git
      - name: Run AI Strategy
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          TV_SESSION_ID: ${{ secrets.TV_SESSION_ID }}
        run: python main.py
