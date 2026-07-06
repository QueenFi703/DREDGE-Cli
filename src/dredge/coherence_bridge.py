"""
CoherenceBridge - bridges coherent data-retrieval and purchasing operations.

This module provides an abstraction layer over an IapService, mirroring the
bridge pattern used in mobile in-app-purchase (IAP) integrations. It follows
the same prepare → execute → report lifecycle that is exercised by the
CoherenceBridgeTest suite.
"""
from __future__ import annotations


class RequestId:
    """Represents an opaque request identifier returned by IAP service calls."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RequestId):
            return self.value == other.value
        return NotImplemented

    def __repr__(self) -> str:
        return f"RequestId({self.value!r})"

    def __hash__(self) -> int:
        return hash(self.value)


class IapService:
    """
    Interface for in-app-purchase service operations.

    Concrete implementations supply the real SDK calls; test doubles (mocks)
    replace individual methods to verify that CoherenceBridge delegates
    correctly.
    """

    def get_user_data(self) -> RequestId:
        """Initiate a user-data request and return its RequestId."""
        raise NotImplementedError

    def get_product_data(self, skus: set[str]) -> RequestId:
        """Initiate a product-data request for *skus* and return its RequestId."""
        raise NotImplementedError

    def purchase(self, sku: str) -> RequestId:
        """Initiate a purchase for *sku* and return its RequestId."""
        raise NotImplementedError

    def get_purchase_updates(self, reset: bool) -> RequestId:
        """
        Retrieve pending purchase updates.

        Parameters
        ----------
        reset:
            When ``True`` the update stream is reset before returning; when
            ``False`` only new updates since the last call are returned.
        """
        raise NotImplementedError


class CoherenceBridgeListener:
    """
    Optional listener that receives asynchronous CoherenceBridge events.

    All methods are no-ops by default so subclasses only need to override the
    callbacks they care about.
    """

    def on_user_data(self, request_id: RequestId) -> None:
        pass

    def on_product_data(self, request_id: RequestId) -> None:
        pass

    def on_purchase_result(self, request_id: RequestId) -> None:
        pass

    def on_purchase_updates(self, request_id: RequestId) -> None:
        pass


class CoherenceBridge:
    """
    Bridge that coordinates coherent data-retrieval and purchasing operations.

    The bridge accepts an :class:`IapService` at construction time and exposes
    a small set of high-level operations that map one-to-one onto the service
    methods.  A :class:`CoherenceBridgeListener` can be attached via
    :meth:`register_listener` (or implicitly during :meth:`prepare_phase`) to
    receive notifications as operations complete.

    Lifecycle
    ---------
    1. *Prepare* – :meth:`prepare_phase` registers the listener and fetches
       user data from the service.
    2. *Query* – :meth:`execute_product_data_query` fetches metadata for one
       or more SKUs.
    3. *Purchase* – :meth:`execute_purchase_phase` initiates a purchase for a
       single SKU.
    4. *Updates* – :meth:`get_purchase_updates` retrieves pending purchase
       updates, optionally resetting the update stream.
    """

    def __init__(self, iap_service: IapService) -> None:
        self._service = iap_service
        self._listener: CoherenceBridgeListener | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_listener(
        self, context: object, listener: CoherenceBridgeListener | None
    ) -> CoherenceBridge:
        """Attach *listener* using the provided *context* and return ``self``."""
        self._listener = listener
        return self

    # ------------------------------------------------------------------
    # Prepare phase
    # ------------------------------------------------------------------

    def prepare_phase(self, context: object) -> RequestId:
        """
        Register the listener with *context* then call ``getUserData``.

        Returns
        -------
        RequestId
            The request identifier returned by the underlying
            :meth:`IapService.get_user_data` call.
        """
        self.register_listener(context, self._listener)
        return self._service.get_user_data()

    # ------------------------------------------------------------------
    # Product-data query
    # ------------------------------------------------------------------

    def execute_product_data_query(self, skus: set[str]) -> RequestId:
        """
        Fetch product data for the supplied set of *skus*.

        Returns
        -------
        RequestId
            The request identifier returned by the underlying
            :meth:`IapService.get_product_data` call.
        """
        return self._service.get_product_data(skus)

    # ------------------------------------------------------------------
    # Purchase phase
    # ------------------------------------------------------------------

    def execute_purchase_phase(self, sku: str) -> RequestId:
        """
        Initiate a purchase for *sku*.

        Returns
        -------
        RequestId
            The request identifier returned by the underlying
            :meth:`IapService.purchase` call.
        """
        return self._service.purchase(sku)

    # ------------------------------------------------------------------
    # Purchase updates
    # ------------------------------------------------------------------

    def get_purchase_updates(self, reset: bool = False) -> RequestId:
        """
        Retrieve pending purchase updates via the :class:`IapService`.

        Parameters
        ----------
        reset:
            Pass ``True`` to reset the update stream, ``False`` to fetch only
            incremental updates since the last call.

        Returns
        -------
        RequestId
            The request identifier returned by the underlying
            :meth:`IapService.get_purchase_updates` call.
        """
        return self._service.get_purchase_updates(reset)
