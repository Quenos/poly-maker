from prometheus_client import Counter, Gauge, Histogram

# Gauges
inventory_g = Gauge("mm_inventory", "Inventory shares per token", ["token_id"])  # type: ignore
delta_usd_g = Gauge("mm_delta_usd", "Delta USD per token", ["token_id"])  # type: ignore
unrealized_g = Gauge("mm_unrealized", "Unrealized PnL per token", ["token_id"])  # type: ignore
spread_g = Gauge("mm_spread", "Current spread", ["token_id"])  # type: ignore
sigma_g = Gauge("mm_sigma", "EWMA sigma", ["token_id"])  # type: ignore

# Counters
fills_c = Counter("mm_fills_total", "Total fills", ["side"])  # type: ignore
cancels_c = Counter("mm_cancels_total", "Total cancels")  # type: ignore
replaces_c = Counter("mm_replaces_total", "Total replaces")  # type: ignore
seq_gaps_c = Counter("mm_seq_gaps_total", "Sequence gaps", ["token_id"])  # type: ignore
reconnects_c = Counter("mm_ws_reconnects_total", "WS reconnects total")  # type: ignore

# Histograms
markout_h = Histogram("mm_markout", "Markout distribution", buckets=(-0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05))  # type: ignore



