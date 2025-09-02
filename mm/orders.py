import random
import time
from typing import List, Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
from py_clob_client.constants import POLYGON
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from mm.state import StateStore, OrderRecord


class OrdersClient:
    """Wrapper around py-clob-client with retries and persistence hooks."""

    def __init__(self, host: str, key: str, funder: str, state: StateStore) -> None:
        self.client = ClobClient(host=host, key=key, chain_id=POLYGON, funder=funder)
        creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds=creds)
        self.state = state

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.1, 1.0))
    def place_order(self, token_id: str, side: str, price: float, size: float, neg_risk: bool = False) -> dict:
        args = OrderArgs(token_id=str(token_id), price=price, size=size, side=side)
        options = PartialCreateOrderOptions(neg_risk=True) if neg_risk else None
        signed = self.client.create_order(args, options=options) if options else self.client.create_order(args)
        resp = self.client.post_order(signed)
        order_id = str(resp.get("order_id") or resp.get("id") or f"p_{int(time.time()*1000)}_{random.randint(1,9999)}")
        self.state.record_order(OrderRecord(order_id=order_id, token_id=token_id, side=side, price=price, size=size, timestamp=time.time()))
        return resp

    def cancel_market_orders(self, market: Optional[str] = None, asset_id: Optional[str] = None) -> None:
        if market:
            self.client.cancel_market_orders(market=market)
        elif asset_id:
            self.client.cancel_market_orders(asset_id=str(asset_id))

    def get_orders(self) -> List[dict]:
        return list(self.client.get_orders())

