const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM} = require('jsdom');
const html = fs.readFileSync('src/dredge/static/studio_billing.html', 'utf8');
const script = fs.readFileSync('src/dredge/static/studio_billing.js', 'utf8');
const tick = () => new Promise(resolve => setTimeout(resolve, 5));
const state = () => ({mode:'sandbox', configured:true, subscribed:false, can_manage:false,
  subscriptions:[], csrf_token:'fixture-token', access:{paid_access:false},
  paypal:{mode:'sandbox', configured:true, subscriptions:[]}, checkout_conflicts:{stripe:false,paypal:false}});
function setup(fetch, url='https://example.test/billing') {
  const dom = new JSDOM(html, {url, runScripts:'outside-only'});
  dom.window.fetch = fetch;
  dom.window.eval(script);
  return dom;
}
const reply = data => ({ok:true, status:200, json:async()=>data});

test('checkout return parameters never activate access or make mutations', async () => {
  const requests=[];
  const dom=setup(async(path, options)=>{requests.push([path,options]); return reply(state());}, 'https://example.test/billing?paypal=returned&subscription_id=I-FOREIGN');
  await tick();
  assert.equal(requests.length,1);
  assert.equal(requests[0][0],'/api/billing/status');
  assert.equal(requests[0][1].method,undefined);
  assert.match(dom.window.document.getElementById('billing-state').textContent,/No confirmed/);
  assert.match(dom.window.document.getElementById('billing-message').textContent,/verified provider/);
  dom.window.close();
});

test('repeated checkout clicks send one CSRF-bound request and reject foreign destinations', async () => {
  let release, posts=0;
  const dom=setup(async(path, options)=>{
    if(!options.method) return reply(state());
    posts++; assert.equal(options.headers['X-CSRF-Token'],'fixture-token');
    return new Promise(resolve=>{release=()=>resolve(reply({url:'https://www.paypal.com.evil.test/approve'}));});
  });
  await tick();
  const button=dom.window.document.getElementById('subscribe-paypal');
  button.click(); button.click();
  assert.equal(posts,1); assert.equal(button.disabled,true);
  assert.equal(dom.window.document.getElementById('subscribe').disabled,true);
  release(); await tick();
  assert.match(dom.window.document.getElementById('billing-message').textContent,/Unexpected billing destination/);
  assert.equal(button.disabled,false);
  dom.window.close();
});

test('cancellation dismissal performs no request; approval refreshes server state', async () => {
  const data=state(); data.paypal.subscriptions=[{status:'active',provider_status:'ACTIVE',cancel_at_period_end:0}];
  let posts=0, approve=false;
  const dom=setup(async(path, options)=>{if(options.method)posts++; return reply(data);});
  dom.window.confirm=()=>approve;
  await tick();
  dom.window.document.getElementById('cancel-paypal').click(); await tick();
  assert.equal(posts,0);
  approve=true;
  dom.window.document.getElementById('cancel-paypal').click(); await tick();
  assert.equal(posts,1);
  assert.match(dom.window.document.getElementById('billing-message').textContent,/refreshed/);
  dom.window.close();
});

test('unconfigured providers and expired sign-in remain disabled', async () => {
  const data=state(); data.configured=false; data.paypal.configured=false;
  let dom=setup(async()=>reply(data)); await tick();
  assert.equal(dom.window.document.getElementById('subscribe').disabled,true);
  assert.equal(dom.window.document.getElementById('subscribe-paypal').disabled,true);
  dom.window.close();
  dom=setup(async()=>({status:401})); await tick();
  assert.equal(dom.window.document.getElementById('billing-signin').hidden,false);
  assert.equal(dom.window.document.getElementById('subscribe-paypal').disabled,true);
  dom.window.close();
});

test('pending PayPal approval can be resumed after returning without paying', async () => {
  const data=state(); data.paypal.subscriptions=[{status:'payment_pending',provider_status:'APPROVAL_PENDING'}];
  data.checkout_conflicts.stripe=true;
  const dom=setup(async()=>reply(data), 'https://example.test/billing?paypal=canceled'); await tick();
  const button=dom.window.document.getElementById('subscribe-paypal');
  assert.equal(button.disabled,false);
  assert.match(button.textContent,/Resume PayPal/);
  assert.equal(dom.window.document.getElementById('subscribe').disabled,true);
  dom.window.close();
});

test('provider refresh is bounded, independent, and does not mask a failure', async () => {
  const paths=[]; const dom=setup(async(path, options)=>{
    if (!options.method) return reply(state());
    paths.push(path);
    if(path==='/api/billing/reconcile') return {ok:false,json:async()=>({error:'Stripe temporarily unavailable.'})};
    return reply({});
  });
  await tick();
  const refresh=dom.window.document.getElementById('refresh-billing'); refresh.click();refresh.click();
  await tick();
  assert.deepEqual(paths,['/api/billing/reconcile','/api/billing/paypal/reconcile']);
  assert.match(dom.window.document.getElementById('billing-message').textContent,/Stripe temporarily unavailable/);
  assert.equal(refresh.disabled,false);
  dom.window.close();
});
