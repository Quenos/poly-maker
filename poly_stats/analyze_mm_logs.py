#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QuoteSample:
    ts: str
    token: str
    market: str
    mid: float
    bid: float
    ask: float


@dataclass
class ComputeSample:
    ts: str
    fair: float
    sigma: float
    h: float
    delta_r: float
    inv_norm: float
    bid: float
    ask: float


QUOTING_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+) .* Quoting token=(?P<token>\d+) market=(?P<market>0x[0-9a-fA-F]+) mid=(?P<mid>\d+\.\d{4}) bid=(?P<bid>\d+\.\d{4}) ask=(?P<ask>\d+\.\d{4})"
)

COMPUTE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} [^ ]+) .* AvellanedaLite compute: fair=(?P<fair>[-\d\.]+) sigma=(?P<sigma>[-\d\.]+) h=(?P<h>[-\d\.]+) delta_r=(?P<dr>[-\d\.]+) inv_norm=(?P<inv>[-\d\.]+) bid=(?P<bid>[-\d\.]+) ask=(?P<ask>[-\d\.]+)"
)


def parse_log(path: str) -> Tuple[List[QuoteSample], List[ComputeSample]]:
    quotes: List[QuoteSample] = []
    computes: List[ComputeSample] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = QUOTING_RE.search(line)
            if m:
                try:
                    quotes.append(
                        QuoteSample(
                            ts=m.group("ts"),
                            token=m.group("token"),
                            market=m.group("market"),
                            mid=float(m.group("mid")),
                            bid=float(m.group("bid")),
                            ask=float(m.group("ask")),
                        )
                    )
                except Exception:
                    pass
                continue
            m2 = COMPUTE_RE.search(line)
            if m2:
                try:
                    computes.append(
                        ComputeSample(
                            ts=m2.group("ts"),
                            fair=float(m2.group("fair")),
                            sigma=float(m2.group("sigma")),
                            h=float(m2.group("h")),
                            delta_r=float(m2.group("dr")),
                            inv_norm=float(m2.group("inv")),
                            bid=float(m2.group("bid")),
                            ask=float(m2.group("ask")),
                        )
                    )
                except Exception:
                    pass
    return quotes, computes


def analyze(quotes: List[QuoteSample], computes: List[ComputeSample]) -> bool:
    if not quotes:
        print("No quoting samples found")
        # Still allow expected to be False due to missing data
        return False
    # Per-token mid stability
    by_token: Dict[str, List[QuoteSample]] = {}
    for q in quotes:
        by_token.setdefault(q.token, []).append(q)
    print(f"Tokens found: {len(by_token)}")
    unstable_tokens = 0
    for token, arr in by_token.items():
        mids = [q.mid for q in arr]
        if not mids:
            continue
        lo = min(mids)
        hi = max(mids)
        stable = (hi - lo) < 0.005
        print(f"Token {token}: samples={len(mids)} mid_min={lo:.4f} mid_max={hi:.4f} stable={stable}")
        if len(mids) > 1 and not stable:
            unstable_tokens += 1
    # Compute sanity checks
    bad_bounds = 0
    bad_sym = 0
    if computes:
        # Check bounds and symmetry between bid/ask around fair
        for c in computes:
            if not (0.01 <= c.bid <= 0.99 and 0.01 <= c.ask <= 0.99):
                bad_bounds += 1
            center = (c.bid + c.ask) / 2.0
            # center should be close to fair + delta_r
            if abs(center - (c.fair + c.delta_r)) > max(0.005, c.h * 0.5):
                bad_sym += 1
        print(f"Compute samples: {len(computes)} bad_bounds={bad_bounds} bad_center_match={bad_sym}")
        # Sigma distribution
        zeros = sum(1 for c in computes if abs(c.sigma) < 1e-9)
        print(f"Sigma zeros: {zeros}/{len(computes)} ({100.0 * zeros/len(computes):.1f}%)")
    # Determine expectedness: no unstable tokens, no compute violations
    expected = (unstable_tokens == 0) and (bad_bounds == 0) and (bad_sym == 0)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MM logs for mids and Avellaneda computes")
    parser.add_argument("--log", default=os.path.abspath(os.path.join(os.getcwd(), "logs", "mm_main.log")))
    args = parser.parse_args()

    path = args.log
    if not os.path.exists(path):
        print(f"Log file not found: {path}")
        sys.exit(1)
    quotes, computes = parse_log(path)
    ok = analyze(quotes, computes)
    if ok:
        print("All values are as expected.")
    else:
        print("Not all values are as expected.")


if __name__ == "__main__":
    main()
