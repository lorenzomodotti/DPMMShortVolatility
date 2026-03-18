WITH TempStocks AS (
  SELECT date, act_symbol, close
  FROM stocks.ohlcv
  WHERE act_symbol = %(ticker)s
),
TempOptions AS (
  SELECT date, act_symbol, call_put, expiration, strike, vol, bid, ask
  FROM options.option_chain
  WHERE act_symbol = %(ticker)s
)
SELECT 
  S.date,
  S.act_symbol,
  S.close AS underlying_price,
  O.call_put,
  O.expiration,
  O.strike,
  O.vol,
  (O.bid + O.ask) / 2 AS option_price
FROM 
  TempStocks AS S
JOIN 
  TempOptions AS O
ON 
  S.date = O.date AND 
  S.act_symbol = O.act_symbol 
ORDER BY
  S.act_symbol, S.date, O.expiration