from prometheus_client import Counter, Gauge

# Only expose new counters/gauges that are not already defined elsewhere to avoid duplicate registration
shock_freezes_total = Counter("mm_shock_freezes_total", "Shock filter freeze events", ["token_id"])  # type: ignore
quote_expired_cancels_total = Counter("mm_quote_expired_cancels_total", "Quote aging cancels", ["token_id"])  # type: ignore
queue_cap_events_total = Counter("mm_queue_cap_events_total", "Queue-aware size cap events", ["token_id", "side"])  # type: ignore
participation_halts_total = Counter("mm_participation_halts_total", "Participation cap halts", ["token_id"])  # type: ignore
backoff_active_gauge = Gauge("mm_backoff_active", "Backoff active multiplier (1=on,0=off)", ["token_id"])  # type: ignore
