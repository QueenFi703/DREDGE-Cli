const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM}=require('jsdom');
test('smoke test consumes consent, blocks double submit, shows failure usage and reloads receipts',async()=>{
 const dom=new JSDOM(fs.readFileSync('src/dredge/static/studio_casework.html','utf8'),{url:'https://example.test/casework',runScripts:'outside-only'});
 const w=dom.window,d=w.document;let resolve,posts=[];
 const receipt={id:'out-1',status:'failed',mode:'live',provider_status:'incomplete',model:'gpt-6-astra',response_id:'resp_1',request_id:'req_1',usage:{input_tokens:80,output_tokens:600},blocks:[],error:'Incomplete; usage retained.',web_search_verified:true,search_calls:[{id:'ws_1',status:'completed',action:{type:'search',sources:['https://dss.mo.gov/']}}]};
 w.fetch=async(path,opts)=>{
  let data={};
  if(path.endsWith('/session'))data={csrf_token:'t',membership:{agency:'a'},encrypted_storage:true};
  else if(path.endsWith('/cases'))data={cases:[]};
  else if(path.endsWith('/pages'))data={pages:[]};
  else if(path.endsWith('/ai')){posts.push(JSON.parse(opts.body));await new Promise(r=>resolve=r);return{ok:false,json:async()=>receipt};}
  else if(path.endsWith('/research-history'))data={outputs:[receipt]};
  return {ok:true,json:async()=>data};
 };
 const until=async fn=>{for(let i=0;i<100;i++){if(fn())return;await new Promise(r=>setTimeout(r,5));}assert.fail('UI timeout');};
 try{
  w.eval(fs.readFileSync('src/dredge/static/studio_casework.js','utf8'));
  await until(()=>d.getElementById('status').textContent.includes('a'));
  const form=d.getElementById('research-smoke');form.elements.consent.checked=true;
  for(let i=0;i<2;i++)form.dispatchEvent(new w.Event('submit',{cancelable:true}));
  await until(()=>resolve);assert.equal(posts.length,1);assert.equal(form.elements.consent.checked,false);assert.equal(posts[0].profile,'public-web-smoke-v1');
  resolve();await until(()=>d.getElementById('receipt-out-1'));
  assert.match(d.getElementById('outputs').textContent,/resp_1.*req_1/);
  assert.match(d.getElementById('outputs').textContent,/Input tokens: 80; output tokens: 600/);
  assert.match(d.getElementById('outputs').textContent,/ws_1.*completed.*search/);
  assert.match(d.getElementById('outputs').textContent,/separate from answer citations/);
  assert.equal(d.querySelector('#outputs a').href,'https://dss.mo.gov/');
  d.getElementById('research-history').click();await until(()=>d.getElementById('message').textContent.startsWith('Loaded'));
  assert.equal(d.querySelectorAll('#receipt-out-1').length,1);
 }finally{w.close();}
});
