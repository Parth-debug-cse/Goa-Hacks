"""Opt-in live integration tests marked with @pytest.mark.live (§3)."""

import os
import pytest
from pom.config import CONFIG


@pytest.mark.live
def test_live_blockchain_rpc_connection():
    if not os.environ.get("POM_RPC_AMOY"):
        pytest.skip("POM_RPC_AMOY not configured")
    from pom.adapters.chain import get_web3_client
    w3 = get_web3_client()
    assert w3.is_connected()
    block = w3.eth.block_number
    assert block > 0
