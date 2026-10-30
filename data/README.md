# Data

**File expected:** `Dataset3_PortfolioReplicaStrategy.xlsx`

Dataset provided as course material; not redistributed.

## Structure

- **Frequency:** weekly
- **Period:** 2007-10-23 → 2021-04-20 (705 observations)
- **Index column:** `Ticker` (dates)

## Columns

| Ticker | Description |
|--------|-------------|
| RX1 | Euro Bund futures (10Y German govt bonds) |
| TY1 | 10Y US Treasury futures |
| GC1 | Gold futures |
| CO1 | Crude Oil (Brent) futures |
| ES1 | S&P 500 E-mini futures |
| VG1 | Euro Stoxx 50 futures |
| NQ1 | Nasdaq 100 futures |
| TP1 | Topix futures |
| DU1 | Schatz futures (2Y German govt bonds) |
| TU2 | 2Y US Treasury futures |
| LLL1 | FTSE 100 futures (excluded — price constant after 2011-06-21) |
| MXWO | MSCI World index |
| MXWD | MSCI World Daily index |
| LEGATRUU | Bloomberg Global Aggregate bond index |
| HFRXGL | HFRX Global Hedge Fund index |

## Usage in notebook

```python
import pandas as pd
from pathlib import Path

data_path = Path("../data/Dataset3_PortfolioReplicaStrategy.xlsx")
df = pd.read_excel(data_path, index_col="Ticker", parse_dates=True)
```
