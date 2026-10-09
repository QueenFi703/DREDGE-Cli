'use strict';
(() => {
  const $=id=>document.getElementById(id);let token=null;
  async function load(){
    const response=await fetch('/api/billing/status',{credentials:'same-origin',headers:{Accept:'application/json'}});
    if(response.status===401){$('billing-state').textContent='Sign in to see your subscription and checkout availability.';$('billing-signin').hidden=false;return;}
    if(!response.ok)throw new Error('Billing status is temporarily unavailable.');
    const data=await response.json();token=data.csrf_token;
    $('billing-mode').textContent=data.mode==='sandbox'?'Sandbox · test payments only. No real charges.':'Live billing · real monthly charges.';
    $('billing-state').textContent=data.configured?(data.subscribed?'Subscription active.':data.subscriptions.length?'Subscription status: '+data.subscriptions.map(s=>s.status.replaceAll('_',' ')).join(', '):'No subscription yet.'):'Checkout is awaiting secure Stripe configuration. No payment will be taken.';
    $('subscribe').textContent=data.mode==='sandbox'?'Test $19/month checkout':'Subscribe for $19/month';
    $('subscribe').disabled=!data.configured||data.subscribed||data.can_manage;
    $('manage').hidden=!data.can_manage;$('manage').disabled=!data.configured;
    if(new URLSearchParams(location.search).get('checkout')==='returned')$('billing-message').textContent='Checkout returned. Payment status is confirmed by Stripe notifications; this page does not activate access.';
  }
  async function open(button,path){button.disabled=true;try{
    const response=await fetch(path,{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:'{}'});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Billing request failed.');
    const target=new URL(data.url);if(target.protocol!=='https:'||!['checkout.stripe.com','billing.stripe.com'].includes(target.hostname))throw new Error('Unexpected billing destination.');
    location.assign(target.href);
  }catch(error){$('billing-message').textContent=error.message;button.disabled=false;}}
  $('subscribe').addEventListener('click',()=>open($('subscribe'),'/api/billing/checkout'));
  $('manage').addEventListener('click',()=>open($('manage'),'/api/billing/portal'));
  load().catch(error=>{$('billing-state').textContent=error.message;});
})();
