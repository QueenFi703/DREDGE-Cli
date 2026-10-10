const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const root = path.join(__dirname,'../src/dredge/static');
const html = fs.readFileSync(path.join(root,'studio_workspace.html'),'utf8');
const script = fs.readFileSync(path.join(root,'studio.js'),'utf8');
const previewRun = {
  id:'public-demo',query:'Fictional library example',status:'completed',mode:'simulated',
  nodes:[
    {id:'frame',dependencies:[],status:'completed',mode:'simulated',duration_ms:null},
    {id:'sources',dependencies:['frame'],status:'completed',mode:'simulated',duration_ms:null},
    {id:'alternatives',dependencies:['frame'],status:'completed',mode:'simulated',duration_ms:null},
    {id:'review',dependencies:['sources','alternatives'],status:'completed',mode:'simulated',duration_ms:null}
  ],
  evidence:[{title:'Example source',url:'https://docs.python.org/3/library/sqlite3.html',excerpt:'<img src=x onerror=alert(1)>',provenance:'demonstration',verification:'unverified'}],
  result:{summary:'Demonstration'},provider_calls:0,token_usage:null,cost_usd:null
};
const report={scope:'Own runs',proposed_runs:0,completed_runs:0,failed_runs:0,success_rate:null,average_duration_ms:null,provider_calls:0,token_usage:null,cost_usd:null,cost_note:'Not metered'};
const status={models:[],tools:[],scope:'Local configuration',persistence:'instance_disk'};
function harness(url,responder){
  const dom=new JSDOM(html,{url,runScripts:'outside-only'});
  dom.window.Element.prototype.scrollIntoView=function(){};
  dom.window.fetch=async(route,options)=>{
    const reply=await responder(route,options);
    return {ok:reply.status>=200&&reply.status<300,status:reply.status,redirected:false,url:String(route),json:async()=>reply.data};
  };
  dom.window.eval(script);
  return dom;
}
async function until(condition){
  for(let i=0;i<100;i++){if(condition())return;await new Promise(resolve=>setTimeout(resolve,5));}
  assert.fail('DOM did not reach the expected state');
}
function signedReplies(role){
  return (route)=>{
    let data;
    if(route==='/api/studio/session')data={name:'Fictional account',role,can_run:role==='operator',can_review:role==='reviewer',csrf_token:'test-token',idle_timeout_seconds:1800,expires_at:Date.now()/1000+3600};
    else if(route==='/api/studio/runs')data={runs:[]};
    else if(route==='/api/studio/status')data=status;
    else if(route==='/api/studio/report')data=report;
    else if(route==='/api/studio/audit')data={events:[],integrity:'Test chain',chain_verified:true};
    else throw new Error('Unexpected route: '+route);
    return {status:200,data};
  };
}
test('guided preview builds a branching graph, safe evidence, and compact view',async()=>{
  const calls=[];const dom=harness('https://studio.example/preview',route=>{calls.push(route);return {status:200,data:previewRun};});
  try{
    const d=dom.window.document;await until(()=>d.querySelectorAll('.graph-node').length===4);
    assert.deepEqual(calls,['/api/studio/preview']);
    assert.equal(d.querySelectorAll('.edge').length,4);
    assert.equal(d.querySelector('#evidence-list img'),null);
    assert.equal(d.querySelector('#evidence-list a').getAttribute('rel'),'noopener noreferrer');
    d.querySelector('.graph-node').dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
    assert.match(d.getElementById('node-detail').textContent,/Timing unavailable/);
    d.getElementById('view-toggle').click();
    assert.equal(d.body.classList.contains('compact'),true);
    assert.equal(d.getElementById('view-toggle').getAttribute('aria-pressed'),'true');
    for(let i=0;i<3;i++)d.getElementById('guide-next').click();
    assert.equal(d.getElementById('tab-reviews').getAttribute('aria-selected'),'true');
    assert.equal(d.getElementById('proposal-details').hidden,true);
  }finally{dom.window.close();}
});
test('viewer navigation hides privileged actions and displays honest empty reports',async()=>{
  const dom=harness('https://studio.example/advanced',signedReplies('viewer'));
  try{const d=dom.window.document;await until(()=>d.getElementById('report-values').children.length===9);
    assert.equal(d.getElementById('proposal-details').hidden,true);
    assert.equal(d.getElementById('tab-audit').hidden,true);
    assert.match(d.getElementById('report-values').textContent,/No samples/);
    assert.match(d.getElementById('report-values').textContent,/Not metered/);
    d.getElementById('tab-execution').dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
    assert.equal(d.getElementById('tab-status').getAttribute('aria-selected'),'true');
    assert.equal(d.getElementById('execution').hidden,true);
    assert.equal(d.getElementById('status').hidden,false);
  }finally{dom.window.close();}
});
test('operator navigation exposes proposal, but not audit or unapproved execution',async()=>{
  const dom=harness('https://studio.example/advanced',signedReplies('operator'));
  try{const d=dom.window.document;await until(()=>d.getElementById('identity').textContent.includes('operator'));
    await until(()=>d.getElementById('workspace-start').hidden===false);
    assert.equal(d.getElementById('start-proposal').hidden,false);
    d.getElementById('start-proposal').click();
    assert.equal(d.getElementById('proposal-details').open,true);
    assert.equal(d.activeElement.id,'run-query');
    assert.equal(d.getElementById('proposal-details').hidden,false);
    assert.equal(d.getElementById('tab-audit').hidden,true);
    assert.equal(d.getElementById('execute-run').hidden,true);
  }finally{dom.window.close();}
});
test('session expiration displays reauthentication and locks mutations',async()=>{
  const dom=harness('https://studio.example/advanced',()=>({status:401,data:{code:'session_expired',error:'Session expired'}}));
  try{const d=dom.window.document;await until(()=>d.getElementById('session-notice').hidden===false);
    assert.equal(d.getElementById('propose-run').disabled,true);
    assert.equal(d.getElementById('start-proposal').disabled,true);
    assert.match(d.getElementById('feedback').textContent,/expired/);
    assert.match(d.getElementById('session-notice').textContent,/Saved runs are retained/);
  }finally{dom.window.close();}
});
test('reviewer decisions require a reason and send the session CSRF token',async()=>{
  let reviewed=false;const posted=[];const run={id:'run-one',query:'Fictional proposal',pipeline_type:'standard',status:'pending_approval'};
  const base=signedReplies('reviewer');
  const dom=harness('https://studio.example/advanced',(route,options)=>{
    if(route==='/api/studio/runs')return {status:200,data:{runs:reviewed?[]:[run]}};
    if(route==='/api/studio/runs/run-one')return {status:200,data:{...run,status:reviewed?'approved':run.status,nodes:[],evidence:[],result:null,can_execute:false}};
    if(route==='/api/studio/runs/run-one/review'){posted.push(options);reviewed=true;return {status:200,data:{status:'approved'}};}
    return base(route);
  });
  try{const d=dom.window.document;await until(()=>d.querySelectorAll('.review-decision').length===2);
    assert.equal(d.getElementById('tab-audit').hidden,false);
    const approve=d.querySelector('.review-decision');approve.click();
    await until(()=>d.getElementById('feedback').textContent.includes('review reason'));
    assert.equal(posted.length,0);
    d.getElementById('review-run-one').value='Ready for local execution.';approve.click();
    await until(()=>posted.length===1);
    assert.equal(posted[0].headers['X-CSRF-Token'],'test-token');
    assert.deepEqual(JSON.parse(posted[0].body),{decision:'approve',note:'Ready for local execution.'});
    await until(()=>d.getElementById('review-list').textContent.includes('No requests are awaiting approval.'));
    await new Promise(resolve=>setTimeout(resolve,0));
  }finally{dom.window.close();}
});

test('disconnected providers are never labeled live in status badges',async()=>{
  const base=signedReplies('viewer');
  const dom=harness('https://studio.example/advanced',(route,options)=>route==='/api/studio/status'?{status:200,data:{models:[{name:'External AI',status:'not_connected',mode:'live'}],tools:[],scope:'Local review',persistence:'configured_path'}}:base(route,options));
  try{const d=dom.window.document;await until(()=>d.querySelector('#status-list .badge'));
    assert.equal(d.querySelector('#status-list .badge').textContent,'Disconnected');
    assert.equal(d.getElementById('start-proposal').hidden,true);
    assert.match(d.getElementById('start-description').textContent,/viewer role/);
  }finally{dom.window.close();}
});

test('audit integrity distinguishes failed, unknown and verified chains on refresh',async()=>{
  const base=signedReplies('reviewer');let verified=false;
  const dom=harness('https://studio.example/advanced',(route,options)=>route==='/api/studio/audit'?{status:200,data:{events:[],integrity:'Hash-linked records.',chain_verified:verified}}:base(route,options));
  try{
    const d=dom.window.document;await until(()=>d.getElementById('audit-integrity').textContent.includes('verification failed'));
    const notice=d.getElementById('audit-integrity');
    assert.equal(notice.getAttribute('role'),'alert');
    assert.equal(notice.classList.contains('error'),true);
    assert.match(notice.textContent,/Do not rely/);
    verified=undefined;d.getElementById('refresh-audit').click();
    await until(()=>notice.textContent.includes('unavailable'));
    assert.equal(notice.classList.contains('error'),true);
    verified=true;d.getElementById('refresh-audit').click();
    await until(()=>notice.textContent.includes('verified'));
    assert.equal(notice.getAttribute('role'),'status');
    assert.equal(notice.classList.contains('error'),false);
  }finally{dom.window.close();}
});

test('live proposals disclose provider, require both confirmations, and reset consent',async()=>{
  const base=signedReplies('operator');const posts=[];
  const run={id:'live-one',query:'Fictional question',pipeline_type:'standard',mode:'live',model:'gpt-6-astra',status:'pending_approval',nodes:[],evidence:[],can_execute:true};
  const dom=harness('https://studio.example/advanced',(route,options)=>{
    if(route==='/api/studio/session') return {status:200,data:{...base(route).data,live_configured:true,live_model:'gpt-6-astra'}};
    if(route==='/api/studio/runs' && options.method==='POST'){posts.push(JSON.parse(options.body));return {status:201,data:run};}
    return base(route);
  });
  try{
    const d=dom.window.document;await until(()=>d.getElementById('live-config').textContent.includes('gpt-6-astra'));
    assert.equal(d.getElementById('run-mode').value,'live');
    assert.equal(d.getElementById('run-pipeline').disabled,true);
    d.getElementById('run-query').value='Fictional question';
    d.getElementById('proposal-form').dispatchEvent(new dom.window.Event('submit',{cancelable:true}));
    await until(()=>d.getElementById('feedback').textContent.includes('Confirm'));
    assert.equal(posts.length,0);
    d.getElementById('provider-consent').checked=true;d.getElementById('public-confirmed').checked=true;
    d.getElementById('source-excerpt').dispatchEvent(new dom.window.Event('input'));
    assert.equal(d.getElementById('provider-consent').checked,false);
    assert.equal(d.getElementById('public-confirmed').checked,false);
    d.getElementById('provider-consent').checked=true;d.getElementById('public-confirmed').checked=true;
    d.getElementById('proposal-form').dispatchEvent(new dom.window.Event('submit',{cancelable:true}));
    await until(()=>posts.length===1);
    assert.deepEqual(posts[0],{query:'Fictional question',pipeline_type:'standard',evidence:[],execution_mode:'live',consent:true,public_data_confirmed:true});
    await until(()=>d.getElementById('feedback').textContent.includes('Proposal saved'));
    assert.equal(d.getElementById('provider-consent').checked,false);
    d.getElementById('run-mode').value='demo';d.getElementById('run-mode').dispatchEvent(new dom.window.Event('change'));
    assert.equal(d.getElementById('live-consent').hidden,true);
    assert.equal(d.getElementById('run-pipeline').disabled,false);
  }finally{dom.window.close();}
});

test('live answer, response trace, partial usage, and unknown source warning render safely',async()=>{
  const base=signedReplies('operator');
  const run={id:'live-done',query:'Fictional',status:'completed',mode:'live',model:'gpt-6-astra',token_usage:120,
    nodes:[{id:'astra_response',dependencies:[],mode:'live',status:'completed',duration_ms:12,response_id:'resp_test',model:'gpt-6-astra',usage:{total_tokens:120}}],evidence:[],
    result:{answer:'<img src=x onerror=alert(1)> [source-99]',provider:'OpenAI',model:'gpt-6-astra',response_id:'resp_test',evidence_review:{unknown_source_ids:['source-99']}}};
  const dom=harness('https://studio.example/advanced',(route,options)=>{
    if(route==='/api/studio/runs')return {status:200,data:{runs:[run]}};
    if(route==='/api/studio/runs/live-done')return {status:200,data:run};
    if(route==='/api/studio/report')return {status:200,data:{...report,provider_calls:2,token_usage:120,usage_status:'partial',calls_without_usage:1}};
    return base(route,options);
  });
  try{
    const d=dom.window.document;await until(()=>d.getElementById('answer-panel').hidden===false);
    assert.equal(d.querySelector('#answer-panel img'),null);
    assert.match(d.getElementById('answer-text').textContent,/<img/);
    assert.match(d.getElementById('provider-record').textContent,/resp_test/);
    assert.match(d.getElementById('reference-warning').textContent,/unknown source IDs: source-99/);
    assert.match(d.getElementById('report-values').textContent,/120 \(partial; 1 calls missing usage\)/);
    d.querySelector('.graph-node').dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true}));
    assert.match(d.getElementById('node-detail').textContent,/resp_test/);
    assert.match(d.getElementById('trace-context').textContent,/Live API/);
  }finally{dom.window.close();}
});

test('saved-run link opens the requested proposal instead of the newest run',async()=>{
  const base=signedReplies('reviewer');
  const run={id:'eac0654f-52ab-4a0e-89e2-b53ac9491308',query:'Exact requested fictional proposal',status:'pending_approval',mode:'live',nodes:[],evidence:[],result:null};
  const requested=[];
  const dom=harness('https://studio.example/advanced?run='+run.id,(route,options)=>{
    if(route==='/api/studio/runs')return {status:200,data:{runs:[{...run,id:'newest',query:'A different newer run'}]}};
    if(route.startsWith('/api/studio/runs/')){requested.push(route);return {status:200,data:run};}
    return base(route,options);
  });
  try{
    const d=dom.window.document;await until(()=>d.getElementById('run-heading').textContent===run.query);
    assert.deepEqual(requested,['/api/studio/runs/'+run.id]);
    assert.equal(d.getElementById('run-permalink').href,'https://studio.example/advanced?run='+run.id);
    assert.equal(d.getElementById('run-permalink').hidden,false);
    assert.ok(d.getElementById('review-'+run.id));
    d.getElementById('refresh-runs').click();
    await until(()=>requested.length===2);
    await until(()=>d.getElementById('refresh-runs').disabled===false);
    assert.ok(d.getElementById('review-'+run.id));
  }finally{dom.window.close();}
});


test('a delayed linked-run response cannot replace a newer explicit selection',async()=>{
  const base=signedReplies('operator');let resolveLinked;
  const old={id:'older',query:'Emailed proposal',status:'pending_approval',mode:'live',nodes:[],evidence:[],result:null};
  const newer={...old,id:'newer',query:'Newly selected proposal'};
  const dom=harness('https://studio.example/advanced?run=older',(route,options)=>{
    if(route==='/api/studio/runs')return {status:200,data:{runs:[newer]}};
    if(route==='/api/studio/runs/older')return new Promise(resolve=>{resolveLinked=()=>resolve({status:200,data:old});});
    if(route==='/api/studio/runs/newer')return {status:200,data:newer};
    return base(route,options);
  });
  try{
    const d=dom.window.document;await until(()=>resolveLinked);
    d.querySelector('#run-list button').click();
    await until(()=>d.getElementById('run-heading').textContent===newer.query);
    resolveLinked();await new Promise(resolve=>setTimeout(resolve,10));
    assert.equal(d.getElementById('run-heading').textContent,newer.query);
    assert.equal(new URL(dom.window.location.href).searchParams.get('run'),'newer');
  }finally{dom.window.close();}
});


test('expired-session link retains the exact run for server-side sign-in restoration',async()=>{
  const id='eac0654f-52ab-4a0e-89e2-b53ac9491308';
  const dom=harness('https://studio.example/advanced?run='+id,()=>({status:401,data:{error:'Expired'}}));
  try{
    const d=dom.window.document;await until(()=>d.getElementById('session-notice').hidden===false);
    assert.equal(d.querySelector('#session-notice a').href,'https://studio.example/advanced?run='+id);
  }finally{dom.window.close();}
});


test('initialization cannot supersede explicit selection while status is loading',async()=>{
  const base=signedReplies('operator');let resolveStatus,resolveSelected,linkedCalls=0;
  const run={id:'chosen',query:'User selected proposal',status:'pending_approval',mode:'live',nodes:[],evidence:[],result:null};
  const dom=harness('https://studio.example/advanced?run=emailed',(route,options)=>{
    if(route==='/api/studio/runs')return {status:200,data:{runs:[run]}};
    if(route==='/api/studio/status')return new Promise(resolve=>{resolveStatus=()=>resolve(base(route,options));});
    if(route==='/api/studio/runs/chosen')return new Promise(resolve=>{resolveSelected=()=>resolve({status:200,data:run});});
    if(route==='/api/studio/runs/emailed'){linkedCalls++;return {status:200,data:{...run,id:'emailed',query:'Emailed'}};}
    return base(route,options);
  });
  try{
    const d=dom.window.document;await until(()=>resolveStatus);
    d.querySelector('#run-list button').click();await until(()=>resolveSelected);
    resolveStatus();await until(()=>d.getElementById('report-values').children.length===9);
    await new Promise(resolve=>setTimeout(resolve,10));
    assert.equal(linkedCalls,0);resolveSelected();
    await until(()=>d.getElementById('run-heading').textContent===run.query);
    assert.equal(new URL(dom.window.location.href).searchParams.get('run'),'chosen');
  }finally{dom.window.close();}
});
