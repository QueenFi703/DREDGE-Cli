'use strict';
(() => {
  let csrf = '', selected = null, config = null, caseLoad = 0, caseLoading = false;
  const el = id => document.getElementById(id);
  const message = text => { el('message').textContent = text; };
  const sections = ['casework','pages','enterprise'];
  function selectSection(name, focus = false) {
    if (!sections.includes(name)) name='casework';
    for (const section of sections) {
      const active=section===name;
      el(section+'-panel').hidden=!active;
      el(section+'-tab').setAttribute('aria-selected',String(active));
      el(section+'-tab').tabIndex=active?0:-1;
      if(active&&focus)el(section+'-tab').focus();
    }
    if(name==='pages'&&config?.membership&&config?.encrypted_storage)pageList().catch(e=>message(e.message));
  }
  for(const name of sections){
    el(name+'-tab').addEventListener('click',()=>{location.hash=name;selectSection(name);});
    el(name+'-tab').addEventListener('keydown',event=>{
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
      event.preventDefault();
      const index=sections.indexOf(name);
      const next=event.key==='Home'?0:event.key==='End'?sections.length-1:(index+(event.key==='ArrowRight'?1:sections.length-1))%sections.length;
      location.hash=sections[next];selectSection(sections[next],true);
    });
  }
  window.addEventListener('hashchange',()=>selectSection(location.hash.slice(1)));
  selectSection(location.hash.slice(1));

  async function api(path, body) {
    const options = body === undefined ? {} : {method:'POST',headers:{'X-CSRF-Token':csrf},body:body instanceof FormData ? body : JSON.stringify(body)};
    if (body !== undefined && !(body instanceof FormData)) options.headers['Content-Type']='application/json';
    const response = await fetch('/api/casework/'+path, options);
    const data = await response.json();
    if (!response.ok) {if(path==='ai'&&data.id)output(data);throw new Error(data.error || 'Request failed.');}
    return data;
  }
  function action(form, fn) {
    form.addEventListener('submit', async event => {
      event.preventDefault(); const button=form.querySelector('button'); if(button.disabled)return; button.disabled=true; message('Working…');
      try {await fn(form); message('Saved.');} catch(error) {message(error.message);} finally {button.disabled=false;}
    });
  }
  function output(data) {
    if(data.id)document.getElementById('receipt-'+data.id)?.remove();
    const section=document.createElement('section');if(data.id)section.id='receipt-'+data.id;
    const heading=document.createElement('h3'); heading.textContent=`${data.kind || 'Astra'} · ${data.mode} · Human review required`; section.append(heading);
    for (const block of data.blocks || []) {
      const p=document.createElement('p'); p.style.whiteSpace='pre-wrap';
      const chars=Array.from(block.text); let cursor=0;
      const annotations=[...(block.citations || [])].sort((a,b)=>a.start-b.start);
      for (const cite of annotations) {
        if (!Number.isInteger(cite.start) || !Number.isInteger(cite.end) || cite.start<cursor || cite.end>chars.length || cite.end<=cite.start) continue;
        p.append(document.createTextNode(chars.slice(cursor,cite.start).join('')));
        const a=document.createElement('a'); a.href=cite.url; a.target='_blank';a.rel='noopener noreferrer';a.textContent=chars.slice(cite.start,cite.end).join('') || cite.title;p.append(a);cursor=cite.end;
      }
      p.append(document.createTextNode(chars.slice(cursor).join('')));section.append(p);
      for (const cite of annotations) {
        const link=document.createElement('a'); link.href=cite.url;link.textContent=cite.title;link.target='_blank';link.rel='noopener noreferrer'; const row=document.createElement('p');row.append(link);section.append(row);
      }
    }
    for (const source of data.evidence || []) { const link=document.createElement('a'); link.href=source.url;link.textContent='Case evidence '+source.id;const row=document.createElement('p');row.append(link);section.append(row); }
    if(data.id && selected && ['analysis','discernment','legal_draft'].includes(data.kind)){
      const review=document.createElement('button');review.className='secondary';review.textContent='Verify evidence and release reviewed draft';
      const label=document.createElement('label');const checkbox=document.createElement('input');checkbox.type='checkbox';label.append(checkbox,document.createTextNode(data.kind==='legal_draft'?'I completed qualified legal review and checked evidence, citations and client explanations.':'I checked this draft against the evidence and client explanations.'));
      review.addEventListener('click',async()=>{try{if(!checkbox.checked)throw new Error('Complete the review and confirm it before release.');await clientApi(`staff/cases/${selected}/release/${data.id}`,{evidence_verified:true,qualified_legal_review:data.kind==='legal_draft'});message('Reviewed draft released to assigned clients.');}catch(e){message(e.message);}});section.append(label,review);
    }
    if(data.error){const error=document.createElement('p');error.textContent=data.error;section.append(error);}
    const receipt=document.createElement('p');receipt.textContent=`Execution: ${data.status || 'legacy / unreported'} · Provider status: ${data.provider_status || 'unreported'} · Response ID: ${data.response_id || 'unreported'} · Request ID: ${data.request_id || 'unreported'}. Actual billed cost: unavailable.`;section.append(receipt);
    const searches=document.createElement('section');const title=document.createElement('h4');title.textContent='Actual provider search records (separate from answer citations)';searches.append(title);
    const note=document.createElement('p');note.textContent=data.web_search_verified?'A completed web-search call was recorded. This does not verify the accuracy or legal validity of the answer.':'No completed web-search call is evidenced by this receipt. Citations alone do not prove live search.';searches.append(note);
    for(const call of data.search_calls||[]){const p=document.createElement('p');p.textContent=`${call.id || 'ID unreported'} · ${call.status || 'status unreported'} · ${call.action?.type || 'action unreported'}`;searches.append(p);for(const url of [...new Set([call.action?.url,...(call.action?.sources||[])].filter(Boolean))]){if(!/^https:\/\//.test(url))continue;const a=document.createElement('a');a.href=url;a.textContent=url;a.target='_blank';a.rel='noopener noreferrer';const line=document.createElement('p');line.append(a);searches.append(line);}}
    section.append(searches);
    const usage=document.createElement('p');usage.textContent=`Model: ${data.model}. Input tokens: ${data.usage?.input_tokens ?? 'unreported'}; output tokens: ${data.usage?.output_tokens ?? 'unreported'}.`;section.append(usage);el('outputs').prepend(section);
  }
  async function list() {
    const data=await api('cases');el('cases').replaceChildren();
    for(const c of data.cases){const li=document.createElement('li');const b=document.createElement('button');b.textContent=c.title+(c.archived?' · Archived':'');b.addEventListener('click',()=>load(c.id).catch(e=>message(e.message)));li.append(b);el('cases').append(li);}
  }
  async function load(id) {
    const version=++caseLoad;
    caseLoading=true;el('analysis').querySelector('button').disabled=true;
    el('analysis').elements.consent.checked=false;
    el('analysis').elements.include_client_explanations.checked=false;
    try {
    const data=await api('cases/'+id);
    if(version!==caseLoad)return;
    el('analysis').elements.consent.checked=false;
    el('analysis').elements.include_client_explanations.checked=false;
    selected=id;el('case-title').textContent=data.title;el('archive-state').textContent=data.archived?'Archived; retained according to your agency policy.':'';el('files').replaceChildren();el('outputs').replaceChildren();
    for(const f of data.files){const li=document.createElement('li');const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.value=f.id;checkbox.name='evidence';const label=document.createElement('label');label.append(checkbox,document.createTextNode(' '+f.name));const a=document.createElement('a');a.href=`/api/casework/cases/${id}/files/${f.id}`;a.textContent='Download';li.append(label,a);
      if(f.ocr_status && f.ocr_status!=='not_required'){
        checkbox.disabled=f.ocr_status!=='verified';
        const state=document.createElement('p');state.textContent='OCR: '+f.ocr_status.replaceAll('_',' ');li.append(state);
        const review=document.createElement('button');review.type='button';review.textContent=f.ocr_status==='verified'?'View verified text':'Review extracted text';
        review.addEventListener('click',async()=>{try{
          const draft=await api(`cases/${id}/files/${f.id}/text`);const form=document.createElement('form');
          const note=document.createElement('p');note.textContent='Open the original download and check every name, date, amount and page. OCR may omit or misread text. Correct the draft before confirmation.';
          const text=document.createElement('textarea');text.value=draft.text;text.rows=12;text.maxLength=100000;text.setAttribute('aria-label','Extracted text for '+f.name);
          const confirm=document.createElement('input');confirm.type='checkbox';confirm.required=true;const check=document.createElement('label');check.append(confirm,document.createTextNode(' I checked this text against the original document, including page completeness.'));
          const save=document.createElement('button');save.textContent='Save verified text';form.append(note,text,check,save);li.append(form);review.disabled=true;
          action(form,async()=>{await api(`cases/${id}/files/${f.id}/text`,{text:text.value,revision:draft.revision,verified:confirm.checked});await load(id);});
        }catch(e){message(e.message);}});li.append(review);
        if(!['pending_review','verified'].includes(f.ocr_status)){const retry=document.createElement('button');retry.type='button';retry.textContent='Retry local OCR';retry.addEventListener('click',async()=>{retry.disabled=true;try{await api(`cases/${id}/files/${f.id}/ocr`,{});await load(id);}catch(e){message(e.message);retry.disabled=false;}});li.append(retry);}
      }
      el('files').append(li);}
    data.outputs.forEach(output);
    await loadReview(id,data.files);
    } finally {
      if(version===caseLoad){caseLoading=false;el('analysis').querySelector('button').disabled=false;}
    }
  }
  let reviewVersion=0;
  async function loadReview(id, files) {
    const data=await api(`cases/${id}/review`);if(selected!==id)return;reviewVersion=data.version;
    el('record-origin').textContent=`Record origin: ${data.profile.origin.replaceAll('_',' ')} · MACSS: not connected`;
    const profile=el('review-profile');profile.elements.origin.value=data.profile.origin;profile.elements.reference.value=data.profile.reference;profile.elements.purpose.value=data.profile.purpose;profile.elements.authorized.checked=false;
    el('review-source').replaceChildren();const prompt=document.createElement('option');prompt.value='';prompt.textContent='Choose a verified source';el('review-source').append(prompt);
    for(const f of files){if(f.ocr_status&& !['verified','not_required'].includes(f.ocr_status))continue;const o=document.createElement('option');o.value=f.id;o.textContent=f.name;el('review-source').append(o);}
    el('review-timeline').replaceChildren();el('review-event-choices').replaceChildren();
    for(const e of [...data.events].sort((a,b)=>a.date.localeCompare(b.date))){const li=document.createElement('li');li.textContent=`${e.date} · ${e.kind} · ${e.description}${e.amount?' · USD '+e.amount:''} · ${e.locator} `;const source=document.createElement('a');source.href=`/api/casework/cases/${id}/files/${e.file_id}`;source.textContent='Original source';li.append(source);el('review-timeline').append(li);const label=document.createElement('label');const c=document.createElement('input');c.type='checkbox';c.name='review-event-id';c.value=e.id;label.append(c,document.createTextNode(e.date+' · '+e.description));el('review-event-choices').append(label);}
    el('review-issues').replaceChildren();
    for(const issue of data.issues){const section=document.createElement('section');const h=document.createElement('h4');h.textContent=issue.question;const status=document.createElement('p');status.textContent=issue.status.replaceAll('_',' ');section.append(h,status);const f=document.createElement('form');const label=document.createElement('label');label.textContent='Review status';const select=document.createElement('select');for(const value of ['open','clarification_requested','resolved']){const o=document.createElement('option');o.value=value;o.textContent=value.replaceAll('_',' ');select.append(o);}select.value=issue.status;label.append(select);const note=document.createElement('textarea');note.maxLength=2000;note.value=issue.resolution;note.setAttribute('aria-label','Resolution or clarification note');const b=document.createElement('button');b.textContent='Save review status';f.append(label,note,b);action(f,async()=>{await api(`cases/${id}/review`,{action:'resolve',version:reviewVersion,issue_id:issue.id,status:select.value,resolution:note.value});await load(id);});section.append(f);el('review-issues').append(section);}
    const intake=await clientApi(`staff/cases/${id}`);if(selected!==id)return;el('review-explanations').replaceChildren();if(!intake.explanations.length)el('review-explanations').textContent='No client explanation has been submitted.';for(const e of intake.explanations){const p=document.createElement('p');p.textContent=e.text;el('review-explanations').append(p);}
    el('review-packets').replaceChildren();for(const packet of data.packets){const section=document.createElement('section');const p=document.createElement('p');p.textContent='Packet '+packet.id+' · '+packet.status.replaceAll('_',' ');const view=document.createElement('a');view.href=`/api/casework/cases/${id}/packets/${packet.id}`;view.target='_blank';view.rel='noopener';view.textContent='Inspect packet snapshot (JSON)';const inspect=document.createElement('button');inspect.type='button';inspect.textContent='Open review packet';inspect.addEventListener('click',async()=>{try{const detail=await api(`cases/${id}/packets/${packet.id}`);const box=document.createElement('section');const heading=document.createElement('h4');heading.textContent=detail.title+' · '+detail.status.replaceAll('_',' ');box.append(heading);const notice=document.createElement('p');notice.textContent=detail.notice+(detail.matches_current_case?' · Snapshot matches current case.':' · Case changed: prepare a fresh packet before review.');box.append(notice);for(const e of detail.snapshot.review.events){const p=document.createElement('p');p.textContent=e.date+' · '+e.description+' · '+e.locator;box.append(p);}for(const q of detail.snapshot.review.issues){const p=document.createElement('p');p.textContent=q.question+' · '+q.status+' · '+q.resolution;box.append(p);}for(const e of detail.snapshot.client_explanations){const p=document.createElement('p');p.textContent='Client explanation: '+e.text;box.append(p);}for(const f of detail.snapshot.files){const a=document.createElement('a');a.href=`/api/casework/cases/${id}/files/${f.id}`;a.textContent='Original source: '+f.name;const p=document.createElement('p');p.append(a);box.append(p);}for(const draft of detail.snapshot.ai_drafts){const h=document.createElement('h4');h.textContent='Unverified AI draft · '+draft.kind;box.append(h);for(const block of draft.draft.blocks||[]){const p=document.createElement('p');p.textContent=block.text;box.append(p);}}section.append(box);inspect.disabled=true;}catch(e){message(e.message);}});section.append(inspect);const f=document.createElement('form');const label=document.createElement('label');const c=document.createElement('input');c.type='checkbox';c.required=true;label.append(c,document.createTextNode(' I checked the source evidence and client explanations in this snapshot.'));const b=document.createElement('button');b.textContent='Approve as independent reviewer';b.disabled=packet.status==='reviewed';f.append(label,b);action(f,async()=>{await api(`cases/${id}/packets/${packet.id}`,{evidence_verified:c.checked,client_explanations_reviewed:c.checked});await load(id);});section.append(p,view,f);el('review-packets').append(section);}
  }
  action(el('review-profile'),async f=>{if(!selected)throw new Error('Select a case first.');await api(`cases/${selected}/review`,{action:'profile',version:reviewVersion,origin:f.elements.origin.value,reference:f.elements.reference.value,purpose:f.elements.purpose.value,authorized:f.elements.authorized.checked});await load(selected);});
  action(el('review-event'),async f=>{if(!selected)throw new Error('Select a case first.');const body={action:'event',version:reviewVersion};for(const key of ['date','kind','amount','description','file_id','locator'])body[key]=f.elements[key].value;await api(`cases/${selected}/review`,body);f.reset();await load(selected);});
  action(el('review-issue'),async f=>{if(!selected)throw new Error('Select a case first.');await api(`cases/${selected}/review`,{action:'issue',version:reviewVersion,question:f.elements.question.value,event_ids:[...el('review-event-choices').querySelectorAll('input:checked')].map(c=>c.value)});f.reset();await load(selected);});
  el('prepare-packet').addEventListener('click',async()=>{try{if(!selected)throw new Error('Select a case first.');await api(`cases/${selected}/packets`,{});await load(selected);message('Packet prepared. Independent review is required.');}catch(e){message(e.message);}});

  let pageId=null, pageVersion=null;
  function newPage(){pageId=null;pageVersion=null;el('page-form').reset();el('page-form').elements.title.disabled=false;el('page-form').elements.body.disabled=false;el('page-save').disabled=false;el('page-archive').hidden=true;el('page-version').replaceChildren();el('page-heading').textContent='Create a page';el('page-state').textContent='Unsaved page';}
  async function pageList(){const data=await api('pages');el('page-list').replaceChildren();for(const p of data.pages){const li=document.createElement('li');const b=document.createElement('button');b.textContent=p.title;b.addEventListener('click',()=>loadPage(p.id).catch(e=>message(e.message)));li.append(b);el('page-list').append(li);}}
  async function loadPage(id,version){const p=await api('pages/'+id+(version?'?version='+version:''));pageId=id;pageVersion=p.current_version;const form=el('page-form');form.elements.title.value=p.title;form.elements.body.value=p.body;form.elements.title.disabled=!p.can_edit;form.elements.body.disabled=!p.can_edit;el('page-save').disabled=!p.can_edit||p.archived;el('page-archive').hidden=!p.can_edit||p.archived;el('page-heading').textContent=p.title;el('page-state').textContent=`Version ${p.version} of ${p.current_version}${p.can_edit?' · Can edit':' · Read only'}`;el('page-version').replaceChildren();for(const v of p.versions){const o=document.createElement('option');o.value=v.version;o.textContent=`Version ${v.version} · ${new Date(v.created*1000).toLocaleString()}`;o.selected=v.version===p.version;el('page-version').append(o);}}
  el('page-new').addEventListener('click',newPage);
  el('page-version').addEventListener('change',()=>{if(pageId)loadPage(pageId,Number(el('page-version').value)).catch(e=>message(e.message));});
  action(el('page-form'),async form=>{const data=await api(pageId?'pages/'+pageId:'pages',{title:form.elements.title.value,body:form.elements.body.value,expected_version:pageVersion});await pageList();await loadPage(data.id);});
  el('page-archive').addEventListener('click',async()=>{try{await api('pages/'+pageId,{archive:true,expected_version:pageVersion});newPage();await pageList();message('Page archived. Its versions are retained.');}catch(e){message(e.message);}});
  async function clientApi(path,body){const options=body===undefined?{}:{method:'POST',headers:{'X-CSRF-Token':csrf,'Content-Type':'application/json'},body:JSON.stringify(body)};const response=await fetch('/api/client/'+path,options);const data=await response.json();if(!response.ok)throw new Error(data.error||'Client intake request failed.');return data;}
  action(el('client-assign'),async form=>{if(!selected)throw new Error('Select a case first.');await clientApi(`cases/${selected}/assign`,{client_id:form.elements.client_id.value,identity_verified:form.elements.verified.checked});form.reset();});
  el('refresh-intake').addEventListener('click',async()=>{try{if(!selected)throw new Error('Select a case first.');const data=await clientApi(`staff/cases/${selected}`);el('staff-intake').replaceChildren();for(const e of data.explanations){const p=document.createElement('p');p.textContent='Client explanation: '+e.text;el('staff-intake').append(p);}for(const q of data.preparation_requests){const section=document.createElement('section');const p=document.createElement('p');p.textContent=`${q.jurisdiction} · ${q.scope} · ${q.state}`;section.append(p);const form=document.createElement('form');const label=document.createElement('label');label.textContent='Per-case estimate in USD';const input=document.createElement('input');input.type='number';input.min='1';input.max='10000';input.required=true;label.append(input);const button=document.createElement('button');button.textContent='Save preparation estimate';form.append(label,button);form.addEventListener('submit',async event=>{event.preventDefault();try{await clientApi('staff/preparation/'+q.id,{fee_usd:Number(input.value),scope:q.scope});message('Estimate saved. No payment collected.');}catch(e){message(e.message);}});section.append(form);el('staff-intake').append(section);}}catch(e){message(e.message);}});
  action(el('create'),async form=>{const c=await api('cases',{title:form.elements.title.value});await list();await load(c.id);form.reset();});
  action(el('upload'),async form=>{if(!selected)throw new Error('Select a case first.');await api(`cases/${selected}/files`,new FormData(form));await load(selected);form.reset();});
  action(el('quote'),async form=>{const data=await api('enterprise-quote',{agency:form.elements.agency.value,seats:Number(form.elements.seats.value)});form.reset();el('status').textContent=data.message;});
  action(el('analysis'),async form=>{
    if(caseLoading)throw new Error('Wait for the selected case to load.');
    if(!selected)throw new Error('Select a case first.');
    const body={kind:form.elements.kind.value,case_id:selected,file_ids:[...document.querySelectorAll('input[name=evidence]:checked')].map(x=>x.value),question:form.elements.question.value,consent:form.elements.consent.checked,include_client_explanations:form.elements.include_client_explanations.checked};
    // Each attempt consumes its own consent, including a provider error after transfer.
    form.elements.consent.checked=false;form.elements.include_client_explanations.checked=false;
    const data=await api('ai',body);output({...data,kind:body.kind});
  });
  action(el('research'),async form=>{const body={kind:form.elements.kind.value,jurisdiction:form.elements.jurisdiction.value,question:form.elements.question.value,consent:form.elements.consent.checked,public_question_confirmed:form.elements.consent.checked};form.elements.consent.checked=false;const data=await api('ai',body);output({...data,kind:body.kind});});
  action(el('research-smoke'),async form=>{const consent=form.elements.consent.checked;form.elements.consent.checked=false;const data=await api('ai',{kind:'research',profile:'public-web-smoke-v1',question:'Public web connectivity test',consent,public_question_confirmed:consent});output({...data,kind:'Public web smoke test'});});
  el('research-history').addEventListener('click',async()=>{try{const data=await api('research-history');for(const receipt of [...data.outputs].reverse())output(receipt);message('Loaded your recorded public research attempts, including incomplete or failed requests.');}catch(e){message(e.message);}});
  el('archive').addEventListener('click',async()=>{if(!selected)return;try{await api(`cases/${selected}/archive`,{});await list();await load(selected);}catch(e){message(e.message);}});
  (async()=>{try{config=await api('session');csrf=config.csrf_token;el('status').textContent=`${config.membership ? config.membership.agency+' · '+config.membership.role : 'Agency membership not yet assigned'} | Encrypted storage: ${config.encrypted_storage?'ready':'needs configuration'} | Astra: ${config.ai_configured?'configured':'key needed'} | Client data: ${config.real_data_enabled?'enabled':'pending agency approval'}`;if(config.membership&&config.encrypted_storage){await list();await pageList();}}catch(e){message(e.message);}})();
})();
