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
    else if(route==='/api/studio/audit')data={events:[],integrity:'Test chain'};
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
    if(route==='/api/studio/runs/run-one')return {status:200,data:{...run,nodes:[],evidence:[],result:null,can_execute:false}};
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
