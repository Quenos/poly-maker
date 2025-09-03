import time

from mm.orders import OrdersClient, OrdersEngine, DesiredQuote


class MockClient:
    def __init__(self):
        self._orders = []  # dicts with id, token_id, side, price, original_size, size_matched
        self._id = 0

    def create_or_derive_api_creds(self):
        return None

    def set_api_creds(self, creds=None):
        pass

    def get_orders(self):
        return list(self._orders)

    def post_order(self, signed):
        return {"id": signed.get("client_id", f"{int(time.time()*1000)}")}

    def create_order(self, args, options=None):
        self._id += 1
        oid = f"o{self._id}"
        self._orders.append({
            "id": oid,
            "token_id": args.token_id,
            "asset_id": args.token_id,
            "side": args.side,
            "price": args.price,
            "original_size": args.size,
            "size_matched": 0.0,
        })
        return {"client_id": oid}

    def cancel_market_orders(self, asset_id=None, market=None):
        if asset_id:
            self._orders = [o for o in self._orders if str(o.get("asset_id")) != str(asset_id)]


def test_diffing_places_and_cancels():
    mc = MockClient()
    oc = OrdersClient.__new__(OrdersClient)
    oc.client = mc
    oc.state = None
    eng = OrdersEngine(client=oc, tick=0.01, partial_fill_pct=50.0, order_max_age_sec=1, requote_mid_ticks=1, requote_queue_levels=2)

    desired = [
        DesiredQuote(token_id="t1", side="BUY", price=0.45, size=100.0, level=0),
        DesiredQuote(token_id="t1", side="SELL", price=0.55, size=100.0, level=0),
    ]
    actions = eng.sync(desired, mid_by_token={"t1": 0.50})
    assert actions["placed"] and not actions["cancelled"]

    # Change desired to new prices; expect replace
    desired2 = [
        DesiredQuote(token_id="t1", side="BUY", price=0.44, size=100.0, level=0),
        DesiredQuote(token_id="t1", side="SELL", price=0.56, size=100.0, level=0),
    ]
    actions2 = eng.sync(desired2, mid_by_token={"t1": 0.50})
    assert actions2["replaced"]


def test_aged_replace_and_partial_fill():
    mc = MockClient()
    oc = OrdersClient.__new__(OrdersClient)
    oc.client = mc
    oc.state = None
    eng = OrdersEngine(client=oc, tick=0.01, partial_fill_pct=10.0, order_max_age_sec=0, requote_mid_ticks=10, requote_queue_levels=10)

    desired = [DesiredQuote(token_id="t1", side="BUY", price=0.45, size=100.0, level=0)]
    eng.sync(desired, mid_by_token={"t1": 0.50})
    # Mark partial fill on mock
    mc._orders[0]["size_matched"] = 20.0
    actions = eng.sync(desired, mid_by_token={"t1": 0.50})
    assert actions["replaced"] or actions["placed"]



