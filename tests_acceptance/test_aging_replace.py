from mm.orders import OrdersEngine, DesiredQuote


class DummyClient:
    def __init__(self):
        self.orders = []

    def get_orders(self):
        return self.orders

    def place_order(self, token_id, side, price, size):
        oid = f"x{len(self.orders)+1}"
        self.orders.append({"id": oid, "asset_id": token_id, "side": side, "price": price, "original_size": size, "size_matched": 0})
        return {"id": oid}

    def cancel_market_orders(self, asset_id=None, market=None):
        if asset_id:
            self.orders = [o for o in self.orders if o.get("asset_id") != asset_id]


def test_aging_replace():
    dc = DummyClient()
    eng = OrdersEngine(client=dc, tick=0.01, partial_fill_pct=0.0, order_max_age_sec=0, requote_mid_ticks=100, requote_queue_levels=100)
    dq = [DesiredQuote(token_id="t", side="BUY", price=0.45, size=100.0, level=0)]
    eng.sync(dq, mid_by_token={"t": 0.50})
    # Immediate replace due to age==0 rule
    actions = eng.sync(dq, mid_by_token={"t": 0.50})
    assert actions["replaced"] or actions["placed"]



