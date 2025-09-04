import subprocess
import shlex
import logging
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    tx_hash: Optional[str]


def call_merger(amount_6dp: int, condition_id: str, is_neg_risk: bool, timeout_sec: float = 30.0) -> MergeResult:
    """Invoke Node helper to merge neutral bundles.

    amount_6dp: integer USDC in 6 decimals
    condition_id: market condition id (hex)
    is_neg_risk: pass-through flag to merger
    """
    args = [
        "node",
        "poly_merger/merge.js",
        str(int(amount_6dp)),
        str(condition_id),
        "true" if is_neg_risk else "false",
    ]
    cmd = " ".join(shlex.quote(a) for a in args)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        tx_hash = None
        for line in stdout.splitlines():
            if "tx" in line.lower() or "hash" in line.lower():
                tx_hash = line.strip()
                break
        return MergeResult(success=proc.returncode == 0, exit_code=proc.returncode, stdout=stdout, stderr=stderr, tx_hash=tx_hash)
    except subprocess.TimeoutExpired as e:
        logger.warning("Merger timed out for %s: %s", condition_id, e)
        return MergeResult(success=False, exit_code=124, stdout=e.stdout or "", stderr=str(e), tx_hash=None)
    except Exception as e:
        logger.exception("Merger exception for %s", condition_id)
        return MergeResult(success=False, exit_code=1, stdout="", stderr=str(e), tx_hash=None)
