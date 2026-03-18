SELECT date, open, close, low, high
FROM stocks.ohlcv
WHERE act_symbol = %(ticker)s
ORDER BY date