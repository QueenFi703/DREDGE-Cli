'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const controls = ['subscribe', 'manage', 'subscribe-paypal', 'cancel-paypal', 'refresh-billing'];
  let token = null, state = null, busy = false;
  const terminalStripe = subscription => ['canceled', 'incomplete_expired'].includes(subscription.status);
  const terminalPayPal = subscription => ['CANCELLED', 'EXPIRED'].includes(subscription.provider_status);
  function render() {
    if (!state) return;
    const paypal = state.paypal;
    const paypalOpen = paypal.subscriptions.some(s => !terminalPayPal(s));
    const paypalPending = paypal.subscriptions.length > 0 && paypal.subscriptions.every(s =>
      terminalPayPal(s) || s.provider_status === 'APPROVAL_PENDING') &&
      paypal.subscriptions.some(s => s.provider_status === 'APPROVAL_PENDING');
    $('billing-mode').textContent = state.mode === 'sandbox'
      ? 'Stripe sandbox · test payments only. No real charges.' : 'Stripe live billing · real monthly charges.';
    $('paypal-mode').textContent = !paypal.configured ? 'PayPal checkout is awaiting secure configuration.'
      : paypal.mode === 'sandbox' ? 'PayPal sandbox · test payments only. No real charges.'
      : 'PayPal live billing · real monthly charges.';
    const subscriptions = [...state.subscriptions, ...paypal.subscriptions];
    $('billing-state').textContent = state.access.paid_access ? 'Paid subscription recognized for this account.'
      : state.subscribed ? 'Test subscription recognized. Sandbox records do not grant paid access.'
      : subscriptions.length ? 'Subscription status: ' + subscriptions.map(s => s.status.replaceAll('_', ' ')).join(', ')
      : 'No confirmed subscription for this account.';
    $('subscribe').textContent = state.mode === 'sandbox' ? 'Test $19/month Stripe checkout' : 'Subscribe for $19/month with Stripe';
    $('subscribe-paypal').textContent = paypalPending ? 'Resume PayPal checkout'
      : paypal.mode === 'sandbox' ? 'Test $19/month PayPal checkout' : 'Subscribe for $19/month with PayPal';
    $('subscribe').disabled = busy || !state.configured || state.checkout_conflicts.stripe || state.subscriptions.some(s => !terminalStripe(s));
    $('subscribe-paypal').disabled = busy || !paypal.configured || state.checkout_conflicts.paypal || (paypalOpen && !paypalPending);
    $('manage').hidden = !state.can_manage;
    $('manage').disabled = busy || !state.configured;
    $('cancel-paypal').hidden = !paypalOpen;
    $('cancel-paypal').disabled = busy || !paypal.configured;
    $('refresh-billing').hidden = false;
    $('refresh-billing').disabled = busy || (!state.configured && !paypal.configured);
  }
  async function load() {
    const response = await fetch('/api/billing/status', {credentials: 'same-origin', headers: {Accept: 'application/json'}});
    if (response.status === 401) {
      token = null; state = null;
      controls.forEach(id => { $(id).disabled = true; });
      $('billing-state').textContent = 'Sign in to see your subscription and checkout availability.';
      $('billing-signin').hidden = false;
      return;
    }
    if (!response.ok) throw new Error('Billing status is temporarily unavailable.');
    state = await response.json(); token = state.csrf_token;
    $('billing-signin').hidden = true;
    render();
  }
  async function post(path) {
    const response = await fetch(path, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-CSRF-Token': token}, body: '{}'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Billing request failed.');
    return data;
  }
  async function action(path, provider) {
    if (busy || !token) return;
    busy = true; render();
    let navigating = false;
    try {
      const data = await post(path);
      if (provider) {
        const target = new URL(data.url);
        const hosts = provider === 'stripe' ? ['checkout.stripe.com', 'billing.stripe.com']
          : [state.paypal.mode === 'live' ? 'www.paypal.com' : 'www.sandbox.paypal.com'];
        if (target.protocol !== 'https:' || target.username || target.password || target.port || !hosts.includes(target.hostname))
          throw new Error('Unexpected billing destination.');
        navigating = true;
        location.assign(target.href);
      } else {
        await load();
        $('billing-message').textContent = 'Payment provider status refreshed.';
      }
    } catch (error) {
      $('billing-message').textContent = error.message;
      // A failed redirect or uncertain response must not start another checkout automatically.
    } finally {
      if (!navigating) { busy = false; render(); }
    }
  }
  $('subscribe').addEventListener('click', () => action('/api/billing/checkout', 'stripe'));
  $('manage').addEventListener('click', () => action('/api/billing/portal', 'stripe'));
  $('subscribe-paypal').addEventListener('click', () => action('/api/billing/paypal/checkout', 'paypal'));
  $('cancel-paypal').addEventListener('click', () => {
    if (!busy && window.confirm('Cancel future PayPal renewals? Access remains through any verified paid period.'))
      action('/api/billing/paypal/cancel');
  });
  $('refresh-billing').addEventListener('click', async () => {
    if (busy || !token || !state) return;
    busy = true; render();
    try {
      const failures = [];
      for (const [enabled, path] of [[state.configured, '/api/billing/reconcile'], [state.paypal.configured, '/api/billing/paypal/reconcile']]) {
        if (enabled) { try { await post(path); } catch (error) { failures.push(error.message); } }
      }
      await load();
      $('billing-message').textContent = failures.length ? failures.join(' ') : 'Payment provider status refreshed.';
    } catch (error) { $('billing-message').textContent = error.message; }
    finally { busy = false; render(); }
  });
  const query = new URLSearchParams(location.search);
  if (query.get('checkout') === 'returned' || query.get('paypal') === 'returned')
    $('billing-message').textContent = 'Checkout returned. Access depends on verified provider payment status. Use Refresh payment status if it has not appeared yet.';
  load().catch(error => { $('billing-state').textContent = error.message; });
})();
