"""Versioned, random-access exogenous draws independent of policy call counts."""
import hashlib
import json
import numpy as np


def indexed_rng(seed: int, window: int, kind: str, *entities: str) -> np.random.Generator:
    key = json.dumps(["indexed-v1", int(seed), int(window), kind, *entities], separators=(",", ":"))
    entropy = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little")
    return np.random.default_rng(entropy)
