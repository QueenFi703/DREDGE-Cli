'use strict';
(() => {
  let csrf = '', selected = null, config = null;
  const el = id => document.getElementById(id);
  const message = text => { el('message').textContent = text; };
  async function api(path, body) {
    const options = body === undefined ? {} : {method:'POST',headers:{'X-CSRF-Token':csrf},body:body instanceof FormData ? body : JSON.stringify(body)};
    if (body !== undefined && !(body instanceof FormData)) options.headers['Content-Type']='application/json';
    const response = await fetch('/api/casework/'+path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed.');
    return data;
  }
  function action(form, fn) {
    form.addEventListener('submit', async event => {
      event.preventDefault(); const button=form.querySelector('button'); button.disabled=true; message('Working…');
      try {await fn(form); message('Saved.');} catch(error) {message(error.message);} finally {button.disabled=false;}
    });
  }
  function output(data) {
    const section=document.createElement('section');
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
    const usage=document.createElement('p');usage.textContent=`Model: ${data.model}. Input tokens: ${data.usage?.input_tokens ?? 'unreported'}; output tokens: ${data.usage?.output_tokens ?? 'unreported'}.`;section.append(usage);el('outputs').prepend(section);
  }
  async function list() {
    const data=await api('cases');el('cases').replaceChildren();
    for(const c of data.cases){const li=document.createElement('li');const b=document.createElement('button');b.textContent=c.title+(c.archived?' · Archived':'');b.addEventListener('click',()=>load(c.id).catch(e=>message(e.message)));li.append(b);el('cases').append(li);}
  }
  async function load(id) {
    const data=await api('cases/'+id); selected=id;el('case-title').textContent=data.title;el('archive-state').textContent=data.archived?'Archived; retained according to your agency policy.':'';el('files').replaceChildren();el('outputs').replaceChildren();
    for(const f of data.files){const li=document.createElement('li');const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.value=f.id;checkbox.name='evidence';const label=document.createElement('label');label.append(checkbox,document.createTextNode(' '+f.name));const a=document.createElement('a');a.href=`/api/casework/cases/${id}/files/${f.id}`;a.textContent='Download';li.append(label,a);el('files').append(li);}
    data.outputs.forEach(output);
  }
  action(el('create'),async form=>{const c=await api('cases',{title:form.elements.title.value});await list();await load(c.id);form.reset();});
  action(el('upload'),async form=>{if(!selected)throw new Error('Select a case first.');await api(`cases/${selected}/files`,new FormData(form));await load(selected);form.reset();});
  action(el('quote'),async form=>{const data=await api('enterprise-quote',{agency:form.elements.agency.value,seats:Number(form.elements.seats.value)});form.reset();el('status').textContent=data.message;});
  action(el('analysis'),async form=>{if(!selected)throw new Error('Select a case first.');const data=await api('ai',{kind:'analysis',case_id:selected,file_ids:[...document.querySelectorAll('input[name=evidence]:checked')].map(x=>x.value),question:form.elements.question.value,consent:form.elements.consent.checked});output({...data,kind:'Case draft'});});
  action(el('research'),async form=>{const data=await api('ai',{kind:'research',question:form.elements.question.value,consent:form.elements.consent.checked,public_question_confirmed:form.elements.consent.checked});output({...data,kind:'Public research'});});
  el('archive').addEventListener('click',async()=>{if(!selected)return;try{await api(`cases/${selected}/archive`,{});await list();await load(selected);}catch(e){message(e.message);}});
  (async()=>{try{config=await api('session');csrf=config.csrf_token;el('status').textContent=`${config.membership ? config.membership.agency+' · '+config.membership.role : 'Agency membership not yet assigned'} | Encrypted storage: ${config.encrypted_storage?'ready':'needs configuration'} | Astra: ${config.ai_configured?'configured':'key needed'} | Client data: ${config.real_data_enabled?'enabled':'pending agency approval'}`;if(config.membership&&config.encrypted_storage)await list();}catch(e){message(e.message);}})();
})();
