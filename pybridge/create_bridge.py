from typing import Dict
from set_path import append_sys_path

append_sys_path()
import bridge


def create_params(is_dealer_vulnerable: bool = False, is_non_dealer_vulnerable: bool = False,
                  dealer: int = 0, seed: int = 0) -> Dict[str, str]:
    params = {
        "is_dealer_vulnerable": str(is_dealer_vulnerable),
        "is_non_dealer_vulnerable": str(is_non_dealer_vulnerable),
        "dealer": str(dealer),
        "seed": str(seed)
    }
    return params


def create_bridge_game(params=None):
    if params is None:
        params = create_params()
    return bridge.BridgeGame(params)
