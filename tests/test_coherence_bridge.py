"""
Tests for the CoherenceBridge module.

These tests mirror the 11 failing scenarios originally reported in
CoherenceBridgeTest.kt and verify the equivalent Python behaviour:

  - preparePhase passes the correct context to registerListener
  - preparePhase registers listener then calls getUserData
  - preparePhase returns the RequestId from getUserData
  - executeProductDataQuery calls getProductData with the supplied SKUs
  - executeProductDataQuery works with a single SKU
  - executePurchasePhase calls purchase with the correct SKU
  - executePurchasePhase returns the RequestId from purchase
  - getPurchaseUpdates can be called via IapService with reset true
  - getPurchaseUpdates can be called via IapService with reset false
  - full purchase lifecycle - init prepare purchase fulfillment report cleanup
  - full user-data retrieval flow - init prepare report cleanup
"""
from unittest.mock import MagicMock

import pytest

from dredge.coherence_bridge import CoherenceBridge, IapService, RequestId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bridge(service=None):
    """Return a CoherenceBridge wired to *service* (or a fresh MagicMock)."""
    if service is None:
        service = MagicMock(spec=IapService)
    return CoherenceBridge(service), service


# ---------------------------------------------------------------------------
# preparePhase tests
# ---------------------------------------------------------------------------

class TestPreparePhase:
    def test_prepare_phase_passes_correct_context_to_register_listener(self):
        """preparePhase passes the correct context to registerListener."""
        bridge, service = _make_bridge()
        service.get_user_data.return_value = RequestId("uid-1")

        context = {"session": "abc"}

        # Intercept register_listener to capture the context argument.
        captured = {}
        original = bridge.register_listener

        def spy(ctx, listener):
            captured["ctx"] = ctx
            return original(ctx, listener)

        bridge.register_listener = spy
        bridge.prepare_phase(context)

        assert captured["ctx"] is context

    def test_prepare_phase_registers_listener_then_calls_get_user_data(self):
        """preparePhase registers listener then calls getUserData."""
        call_order: list[str] = []
        bridge, service = _make_bridge()

        service.get_user_data.side_effect = (
            lambda: call_order.append("getUserData") or RequestId("uid-2")
        )

        original = bridge.register_listener

        def spy(ctx, listener):
            call_order.append("registerListener")
            return original(ctx, listener)

        bridge.register_listener = spy
        bridge.prepare_phase({})

        assert call_order == ["registerListener", "getUserData"]

    def test_prepare_phase_returns_the_request_id_from_get_user_data(self):
        """preparePhase returns the RequestId from getUserData."""
        expected = RequestId("user-data-request-42")
        bridge, service = _make_bridge()
        service.get_user_data.return_value = expected

        result = bridge.prepare_phase({})

        assert result == expected


# ---------------------------------------------------------------------------
# executeProductDataQuery tests
# ---------------------------------------------------------------------------

class TestExecuteProductDataQuery:
    def test_execute_product_data_query_calls_get_product_data_with_supplied_skus(self):
        """executeProductDataQuery calls getProductData with the supplied SKUs."""
        bridge, service = _make_bridge()
        service.get_product_data.return_value = RequestId("prod-1")

        skus = {"sku-a", "sku-b", "sku-c"}
        bridge.execute_product_data_query(skus)

        service.get_product_data.assert_called_once_with(skus)

    def test_execute_product_data_query_works_with_a_single_sku(self):
        """executeProductDataQuery works with a single SKU."""
        expected = RequestId("prod-single")
        bridge, service = _make_bridge()
        service.get_product_data.return_value = expected

        skus = {"only-sku"}
        result = bridge.execute_product_data_query(skus)

        service.get_product_data.assert_called_once_with(skus)
        assert result == expected


# ---------------------------------------------------------------------------
# executePurchasePhase tests
# ---------------------------------------------------------------------------

class TestExecutePurchasePhase:
    def test_execute_purchase_phase_calls_purchase_with_correct_sku(self):
        """executePurchasePhase calls purchase with the correct SKU."""
        bridge, service = _make_bridge()
        service.purchase.return_value = RequestId("purch-1")

        sku = "premium-upgrade"
        bridge.execute_purchase_phase(sku)

        service.purchase.assert_called_once_with(sku)

    def test_execute_purchase_phase_returns_the_request_id_from_purchase(self):
        """executePurchasePhase returns the RequestId from purchase."""
        expected = RequestId("purch-42")
        bridge, service = _make_bridge()
        service.purchase.return_value = expected

        result = bridge.execute_purchase_phase("any-sku")

        assert result == expected


# ---------------------------------------------------------------------------
# getPurchaseUpdates tests
# ---------------------------------------------------------------------------

class TestGetPurchaseUpdates:
    def test_get_purchase_updates_can_be_called_via_iap_service_with_reset_true(self):
        """getPurchaseUpdates can be called via IapService with reset true."""
        bridge, service = _make_bridge()
        service.get_purchase_updates.return_value = RequestId("upd-reset")

        bridge.get_purchase_updates(reset=True)

        service.get_purchase_updates.assert_called_once_with(True)

    def test_get_purchase_updates_can_be_called_via_iap_service_with_reset_false(self):
        """getPurchaseUpdates can be called via IapService with reset false."""
        bridge, service = _make_bridge()
        service.get_purchase_updates.return_value = RequestId("upd-no-reset")

        bridge.get_purchase_updates(reset=False)

        service.get_purchase_updates.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# Full lifecycle tests
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    def test_full_purchase_lifecycle_init_prepare_purchase_fulfillment_report_cleanup(self):
        """full purchase lifecycle - init prepare purchase fulfillment report cleanup."""
        bridge, service = _make_bridge()

        user_data_id = RequestId("lifecycle-user-data")
        purchase_id = RequestId("lifecycle-purchase")
        updates_id = RequestId("lifecycle-updates")

        service.get_user_data.return_value = user_data_id
        service.purchase.return_value = purchase_id
        service.get_purchase_updates.return_value = updates_id

        context = {"user": "lifecycle-user"}

        # init / prepare
        prepare_result = bridge.prepare_phase(context)
        assert prepare_result == user_data_id

        # purchase
        purchase_result = bridge.execute_purchase_phase("lifecycle-sku")
        assert purchase_result == purchase_id

        # fulfillment report via purchase-updates
        updates_result = bridge.get_purchase_updates(reset=False)
        assert updates_result == updates_id

        # verify delegation
        service.get_user_data.assert_called_once()
        service.purchase.assert_called_once_with("lifecycle-sku")
        service.get_purchase_updates.assert_called_once_with(False)

    def test_full_user_data_retrieval_flow_init_prepare_report_cleanup(self):
        """full user-data retrieval flow - init prepare report cleanup."""
        bridge, service = _make_bridge()

        expected_id = RequestId("flow-user-data")
        service.get_user_data.return_value = expected_id

        context = {"session": "flow-session"}

        # prepare (registers listener + fetches user data)
        result = bridge.prepare_phase(context)

        assert result == expected_id
        service.get_user_data.assert_called_once()
