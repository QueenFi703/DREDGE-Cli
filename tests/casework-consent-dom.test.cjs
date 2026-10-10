const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM}=require('jsdom');

test('explanation sharing is opt-in and resets after a request or case change',async()=>{
  const dom=new JSDOM(fs.readFileSync('src/dredge/static/studio_casework.html','utf8'),{url:'https://example.test/casework',runScripts:'outside-only'});
  const w=dom.window;const posts=[];let delayCase=false,resolveCase,failAi=false;
  w.fetch=async(path,options)=>{
    let data={};
    if(path.endsWith('/session'))data={csrf_token:'token',membership:{agency:'a',role:'caseworker'},encrypted_storage:true};
    else if(path.endsWith('/pages'))data={pages:[]};
    else if(path==='/api/casework/cases')data={cases:[{id:'first',title:'First case'},{id:'second',title:'Second case'}]};
    else if(/^\/api\/casework\/cases\/(first|second)$/.test(path)){
      if(delayCase)await new Promise(resolve=>{resolveCase=resolve;});
      data={title:path.split('/').pop(),files:[{id:'file',name:'Fictional.txt',ocr_status:'not_required'}],outputs:[]};
    }
    else if(path.endsWith('/review'))data={version:0,profile:{origin:'fictional',reference:'',purpose:''},events:[],issues:[],packets:[]};
    else if(path.startsWith('/api/client/'))data={explanations:[],preparation_requests:[]};
    else if(path==='/api/casework/ai'){
      posts.push(JSON.parse(options.body));
      if(failAi)return {ok:false,json:async()=>({error:'Provider failed after receiving request.'})};
      data={blocks:[],usage:{}};
    }
    return {ok:true,json:async()=>data};
  };
  const until=async fn=>{for(let i=0;i<100;i++){if(fn())return;await new Promise(r=>setTimeout(r,5));}assert.fail('Expected UI not reached');};
  try{
    w.eval(fs.readFileSync('src/dredge/static/studio_casework.js','utf8'));
    await until(()=>w.document.querySelectorAll('#cases button').length===2);
    const d=w.document;d.querySelector('#cases button').click();
    await until(()=>d.querySelector('#files input'));
    const form=d.getElementById('analysis');const choice=form.elements.include_client_explanations;
    assert.ok(choice);assert.equal(choice.checked,false);
    assert.match(choice.parentElement.textContent,/latest five client explanations.*OpenAI/);
    form.elements.question.value='Summarize fictional text';form.elements.consent.checked=true;
    d.querySelector('#files input').checked=true;
    form.dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
    await until(()=>posts.length===1&&!form.querySelector('button').disabled);
    assert.equal(posts[0].include_client_explanations,false);
    assert.equal(form.elements.consent.checked,false);
    choice.checked=true;form.elements.consent.checked=true;
    form.dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
    await until(()=>posts.length===2&&!form.querySelector('button').disabled);
    assert.equal(posts[1].include_client_explanations,true);
    assert.equal(choice.checked,false);
    delayCase=true;choice.checked=true;form.elements.consent.checked=true;
    d.querySelectorAll('#cases button')[1].click();
    await until(()=>resolveCase);
    assert.equal(form.querySelector('button').disabled,true);
    choice.checked=true;form.elements.consent.checked=true;
    resolveCase();
    await until(()=>d.getElementById('case-title').textContent==='second'&&!form.querySelector('button').disabled);
    assert.equal(choice.checked,false);assert.equal(form.elements.consent.checked,false);
    failAi=true;choice.checked=true;form.elements.consent.checked=true;
    d.querySelector('#files input').checked=true;
    form.dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
    await until(()=>posts.length===3&&!form.querySelector('button').disabled);
    assert.equal(posts[2].include_client_explanations,true);
    assert.equal(posts[2].case_id,'second');
    assert.equal(choice.checked,false);assert.equal(form.elements.consent.checked,false);
    // A delayed older case response must not replace the latest selected case.
    resolveCase=null;d.querySelectorAll('#cases button')[0].click();
    await until(()=>resolveCase);const firstResponse=resolveCase;
    resolveCase=null;d.querySelectorAll('#cases button')[1].click();
    await until(()=>resolveCase);const secondResponse=resolveCase;
    secondResponse();
    await until(()=>!form.querySelector('button').disabled);
    choice.checked=true;form.elements.consent.checked=true;
    firstResponse();await new Promise(resolve=>setTimeout(resolve,10));
    assert.equal(d.getElementById('case-title').textContent,'second');
    assert.equal(choice.checked,true);assert.equal(form.elements.consent.checked,true);
    assert.equal(posts.length,3);
  }finally{dom.window.close();}
});
