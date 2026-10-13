const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM} = require('jsdom');
const html = fs.readFileSync('src/dredge/static/studio_research_controls.html', 'utf8');
const flush = () => new Promise(resolve => setTimeout(resolve, 10));
function page() {
  const calls = [], pending=[];
  const dom = new JSDOM(html, {url:'https://dredge.example/studio/research-controls',runScripts:'dangerously',beforeParse(window){window.fetch=async(path,options)=>{
    calls.push([path,options]);
    if(path.endsWith('/session')) return {ok:true,json:async()=>({csrf_token:'fixture-csrf',role:'operator'})};
    if(path.endsWith('/budget')) return {ok:true,json:async()=>({enabled:false,limit_micro_usd:30000000,held_micro_usd:0,committed_upper_micro_usd:0,available_micro_usd:30000000,pricing:{status:'current'}})};
    return new Promise(resolve=>pending.push(data=>resolve({ok:true,json:async()=>data})));
  };}});
  return {dom,calls,pending,document:dom.window.document};
}
function submit(p, id) {
  p.document.getElementById('base-id').value='original';p.document.getElementById('new-id').value=id;
  p.document.getElementById('compare-form').dispatchEvent(new p.dom.window.Event('submit',{bubbles:true,cancelable:true}));
}
test('comparison sends only selected run pair with CSRF and renders untrusted text safely',async()=>{
  const p=page(); await flush(); submit(p,'newer');
  const sent=p.calls.find(([path])=>path.endsWith('/refresh-plan'));
  assert.deepEqual(JSON.parse(sent[1].body),{base_run_id:'original',new_run_id:'newer'});
  assert.equal(sent[1].headers['X-CSRF-Token'],'fixture-csrf');
  p.pending[0]({sections:[{text:'<img src=x onerror=alert(1)>'}]}); await flush();
  assert.match(p.document.getElementById('compare-result').textContent,/img src/);
  assert.equal(p.document.querySelectorAll('img').length,0);
  assert.equal(p.document.getElementById('cap-form'),null);
  p.dom.window.close();
});
test('older comparison response cannot overwrite newest selected pair',async()=>{
  const p=page();await flush();submit(p,'new-A');submit(p,'new-B');
  p.pending[1]({new_run_id:'new-B'});await flush();p.pending[0]({new_run_id:'new-A'});await flush();
  assert.match(p.document.getElementById('compare-result').textContent,/new-B/);
  assert.doesNotMatch(p.document.getElementById('compare-result').textContent,/new-A/);
  p.dom.window.close();
});
test('editing inputs clears old results and suppresses in-flight response',async()=>{
  const p=page();await flush();submit(p,'new-A');
  p.document.getElementById('new-id').value='new-B';
  p.document.getElementById('new-id').dispatchEvent(new p.dom.window.Event('input'));
  p.pending[0]({new_run_id:'new-A'});await flush();
  assert.equal(p.document.getElementById('compare-result').textContent,'');
  p.dom.window.close();
});
