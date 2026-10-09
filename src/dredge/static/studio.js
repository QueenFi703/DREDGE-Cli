'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const preview = location.pathname === '/preview';
  let identity = null, selected = null, runs = [], expired = false, guideStep = 0;
  let lastActivity = Date.now(), idleTimer = null;
  const labels = {pending_approval:'Awaiting approval', approved:'Approved', rejected:'Rejected', running:'Running', completed:'Completed', failed:'Failed'};
  const el = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; };
  const badge = (text, mode) => el('span', text, 'badge ' + (mode || ''));
  const say = message => { $('feedback').textContent = message; };
  function lockSession() {
    expired = true; $('session-notice').hidden = false;
    $('session-notice').scrollIntoView({block:'nearest'});
    document.querySelectorAll('form button, #execute-run, #start-proposal, .review-decision').forEach(node => { node.disabled = true; });
    say('Your session expired. Sign in again to continue.');
  }
  async function api(path, options = {}) {
    if (expired && !preview) throw new Error('Sign in again to continue.');
    const response = await fetch(path, {credentials:'same-origin', headers:{'Accept':'application/json', 'Content-Type':'application/json', ...(identity ? {'X-CSRF-Token':identity.csrf_token} : {})}, ...options});
    if (response.status === 401 || (response.redirected && response.url.includes('/auth/login'))) { lockSession(); throw new Error('Your session expired.'); }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  async function action(button, work) {
    button.disabled = true;
    try { await work(); } catch (error) { say(error.message); }
    finally { if (!expired) button.disabled = false; }
  }
  function selectTab(name, focus = false) {
    document.querySelectorAll('[data-tab]').forEach(button => {
      const active = button.dataset.tab === name;
      button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1;
      $(button.dataset.tab).hidden = !active;
      if (active && focus) button.focus();
    });
  }
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.addEventListener('click', () => selectTab(button.dataset.tab));
    button.addEventListener('keydown', event => {
      const tabs = [...document.querySelectorAll('[data-tab]')].filter(tab => !tab.hidden);
      const index = tabs.indexOf(button);
      let next = null;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== null) { event.preventDefault(); selectTab(tabs[next].dataset.tab, true); }
    });
  });
  let compact = false;
  try { compact = localStorage.getItem('dredge.studio.compact') === 'true'; } catch (_) {}
  function setCompact(value) {
    compact = value; document.body.classList.toggle('compact', compact);
    $('view-toggle').textContent = compact ? 'Expanded view' : 'Compact view';
    $('view-toggle').setAttribute('aria-pressed', String(compact));
  }
  setCompact(compact);
  $('view-toggle').addEventListener('click', () => { setCompact(!compact); try { localStorage.setItem('dredge.studio.compact', String(compact)); } catch (_) {} });

  function inspectNode(node) {
    $('node-detail').hidden = false;
    $('node-detail').replaceChildren(el('strong', node.id), el('p', `${node.mode} · ${node.status} · ${node.duration_ms == null ? 'Timing unavailable' : node.duration_ms + ' ms measured on the server'}`), el('p', 'Dependencies: ' + (node.dependencies.join(', ') || 'None')));
  }
  function drawGraph(nodes) {
    $('graph').replaceChildren(); $('node-list').replaceChildren(); $('node-detail').hidden = true;
    if (!nodes.length) { $('graph').append(el('p','No node events recorded yet. Approval precedes execution.','muted')); return; }
    const levels = new Map(); const byId = new Map(nodes.map(node => [node.id, node]));
    function level(id, visiting = new Set()) {
      if (levels.has(id)) return levels.get(id);
      if (visiting.has(id) || !byId.has(id)) return 0;
      const next = new Set(visiting); next.add(id);
      const dependencies = byId.get(id).dependencies || [];
      const value = dependencies.length ? 1 + Math.max(...dependencies.map(dep => level(dep, next))) : 0;
      levels.set(id, value); return value;
    }
    nodes.forEach(node => level(node.id));
    const buckets = [];
    nodes.forEach(node => { const depth = levels.get(node.id); (buckets[depth] ||= []).push(node); });
    const width = 620, rowHeight = 108, nodeWidth = 180;
    const positions = new Map(); let row = 0;
    buckets.forEach(bucket => {
      for (let i = 0; i < bucket.length; i += 3) {
        const group = bucket.slice(i, i + 3);
        group.forEach((node,index) => positions.set(node.id,{x:(width/group.length)*(index+.5)-nodeWidth/2,y:row*rowHeight+15})); row++;
      }
    });
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns,'svg'); svg.setAttribute('viewBox',`0 0 ${width} ${row*rowHeight}`); svg.setAttribute('role','group'); svg.setAttribute('aria-label','Execution graph; select a node for recorded details');
    nodes.forEach(node => {
      const target = positions.get(node.id);
      (node.dependencies || []).forEach(dependency => {
        const source = positions.get(dependency); if (!source) return;
        const edge = document.createElementNS(ns,'path'); edge.setAttribute('class','edge');
        edge.setAttribute('d',`M ${source.x+nodeWidth/2} ${source.y+72} L ${target.x+nodeWidth/2} ${target.y}`); svg.append(edge);
      });
    });
    nodes.forEach(node => {
      const position = positions.get(node.id); const group = document.createElementNS(ns,'g');
      group.setAttribute('class','graph-node'); group.setAttribute('role','button'); group.setAttribute('tabindex','0');
      group.setAttribute('aria-label',`${node.id}, ${node.status}, ${node.mode}`); group.setAttribute('transform',`translate(${position.x},${position.y})`);
      const rect = document.createElementNS(ns,'rect'); rect.setAttribute('width',String(nodeWidth)); rect.setAttribute('height','72'); rect.setAttribute('rx','12'); group.append(rect);
      const title = document.createElementNS(ns,'text'); title.setAttribute('x','12'); title.setAttribute('y','26'); title.setAttribute('font-size','13'); title.textContent=node.id; group.append(title);
      const state = document.createElementNS(ns,'text'); state.setAttribute('x','12'); state.setAttribute('y','50'); state.setAttribute('font-size','10'); state.textContent=`${node.mode} · ${node.status}`; group.append(state);
      group.addEventListener('click',()=>inspectNode(node)); group.addEventListener('keydown',event=>{ if (event.key==='Enter'||event.key===' ') {event.preventDefault();inspectNode(node);} }); svg.append(group);
      const button = el('button',`${node.id} · ${node.status}`,'secondary'); button.addEventListener('click',()=>inspectNode(node)); $('node-list').append(button);
    });
    $('graph').append(svg);
  }
  function renderEvidence(sources) {
    $('evidence-list').replaceChildren();
    if (!sources.length) { $('evidence-list').append(el('p','No source records attached to this run.','muted')); return; }
    sources.forEach(source => {
      const article = el('article',undefined,'source-record'); article.append(el('h3',source.title));
      try { const url = new URL(source.url); if (url.protocol === 'https:' && !url.username && !url.password) { const link = el('a',url.hostname + ' ↗'); link.href=url.href; link.target='_blank'; link.rel='noopener noreferrer'; article.append(link); } } catch (_) {}
      const meta=el('div',undefined,'source-meta'); meta.append(badge(source.provenance),badge(source.verification)); article.append(meta);
      article.append(el('blockquote',source.excerpt || 'No excerpt was supplied.')); $('evidence-list').append(article);
    });
  }
  function renderRun(run) {
    selected = run; $('workspace-start').hidden=true; $('graph').hidden=false; $('run-heading').textContent=run.query; $('run-state').textContent=labels[run.status]||run.status; $('run-state').className='badge '+run.status;
    $('trace-context').textContent=preview ? 'Public demonstration graph. No measured timings or live execution.' : `Run ${run.id} · Actual stored local DAG events; simulated steps retain their own labels.`;
    drawGraph(run.nodes||[]); renderEvidence(run.evidence||[]);
    $('run-result').textContent=run.result ? JSON.stringify(run.result,null,2) : 'No result recorded.';
    $('execute-run').hidden=preview || !run.can_execute || run.status!=='approved';
    renderRunList();
  }
  function renderRunList() {
    $('run-list').replaceChildren();
    if (!runs.length) $('run-list').append(el('p','No saved runs yet.','muted'));
    if (!preview) {
      $('workspace-start').hidden=!!runs.length;
      if (!runs.length) {
        $('graph').hidden=true;
        $('run-heading').textContent='Your first run starts here';
        $('trace-context').textContent='No recorded execution yet.';
        $('start-proposal').hidden=!identity?.can_run;
        $('start-heading').textContent=identity?.can_run?'Frame a question. Propose the work.':'Explore how DREDGE works';
        $('start-description').textContent=identity?.can_run?'Create a local pipeline proposal with optional source records. A different reviewer must approve it before you can execute.':'Your viewer role lets you inspect your saved work. Try the public preview now; to create runs, the administrator must assign your OAuth account an operator role and arrange a separate reviewer.';
      }
    }
    runs.forEach(run=>{
      const button=el('button',`${run.query} — ${labels[run.status]||run.status}`,'secondary'); button.setAttribute('aria-pressed',String(selected?.id===run.id));
      button.addEventListener('click',()=>action(button,async()=>{renderRun(await api('/api/studio/runs/'+encodeURIComponent(run.id)));})); $('run-list').append(button);
    });
  }
  $('start-proposal').addEventListener('click',()=>{if(!identity?.can_run||expired)return;selectTab('execution');$('proposal-details').open=true;$('run-query').focus();});
  async function refreshRuns() { const data=await api('/api/studio/runs'); runs=data.runs; renderRunList(); renderReviews(); }
  function renderReviews() {
    $('review-list').replaceChildren();
    const pending = runs.filter(run=>run.status==='pending_approval');
    if (!pending.length) $('review-list').append(el('p','No requests are awaiting approval.','muted'));
    pending.forEach(run=>{
      const article=el('article',undefined,'review-record'); article.append(el('h3',run.query),el('p',`${run.pipeline_type} · ${run.id}`,'muted small'));
      const inspect=el('button','Inspect proposal & sources','secondary'); inspect.addEventListener('click',()=>action(inspect,async()=>{renderRun(await api('/api/studio/runs/'+encodeURIComponent(run.id)));selectTab('execution',true);})); article.append(inspect);
      if (identity?.can_review) {
        const label=el('label','Review reason'); const textarea=el('textarea'); textarea.id='review-'+run.id; textarea.maxLength=2000; textarea.rows=2; label.htmlFor=textarea.id; article.append(label,textarea);
        const actions=el('div',undefined,'actions');
        ['approve','reject'].forEach(decision=>{const button=el('button',decision==='approve'?'Approve':'Reject','secondary review-decision'); button.addEventListener('click',()=>action(button,async()=>{if (!textarea.value.trim()) throw new Error('Enter a review reason.');await api(`/api/studio/runs/${encodeURIComponent(run.id)}/review`,{method:'POST',body:JSON.stringify({decision,note:textarea.value})});say('Review recorded.');await refreshRuns();}));actions.append(button);}); article.append(actions);
      }
      $('review-list').append(article);
    });
    $('review-note').textContent=identity?.can_review ? 'Reviewers can inspect instance proposals. The server prevents self-approval and duplicate decisions.' : 'Your role can view your requests. A separate reviewer or admin must approve them.';
  }
  function renderStatus(data) {
    $('status-list').replaceChildren();
    [...data.models,...data.tools].forEach(item=>{const row=el('div',undefined,'status-record');row.append(el('strong',item.name),el('p',item.status.replaceAll('_',' '),'muted'),badge(item.status==='not_connected'?'Disconnected':item.mode,item.status==='not_connected'?'disconnected':item.mode));if(item.last_observed)row.append(el('p','Observed: '+new Date(item.last_observed*1000).toLocaleString(),'small'));$('status-list').append(row);});
    $('status-scope').textContent=`${data.scope} Storage: ${data.persistence.replaceAll('_',' ')}.`;
  }
  async function refreshStatus(){renderStatus(await api('/api/studio/status'));}
  function renderReport(data) {
    $('report-scope').textContent=data.scope; $('report-values').replaceChildren();
    const metrics=[['Proposed runs',data.proposed_runs],['Completed',data.completed_runs],['Failed',data.failed_runs],['Success rate',data.success_rate==null?'No samples':(data.success_rate*100).toFixed(1)+'%'],['Mean local duration',data.average_duration_ms==null?'No samples':data.average_duration_ms+' ms'],['External calls',data.provider_calls],['Tokens',data.token_usage??'Unavailable'],['Cost',data.cost_usd==null?'Not metered':'$'+data.cost_usd.toFixed(4)],['Interrupted / running',data.interrupted_runs??0]];
    metrics.forEach(([name,value])=>{const row=el('div');row.append(el('dt',name),el('dd',String(value)));$('report-values').append(row);}); $('cost-note').textContent=data.cost_note;
  }
  async function refreshReport(){renderReport(await api('/api/studio/report'));}
  async function refreshAudit(){const data=await api('/api/studio/audit');$('audit-integrity').textContent=data.integrity;$('audit-list').replaceChildren();if(!data.events.length)$('audit-list').append(el('p','No audit events recorded.','muted'));data.events.forEach(event=>{const row=el('article',undefined,'audit-record');row.append(el('strong',event.action.replaceAll('_',' ')),el('p',`${new Date(event.recorded*1000).toLocaleString()} · ${event.actor}`,'muted small'),el('p','Run: '+(event.run_id||'—'),'small'));const details=el('details');details.append(el('summary','Integrity record'),el('pre',JSON.stringify({seq:event.seq,detail:JSON.parse(event.detail),hash:event.hash,previous_hash:event.previous_hash},null,2)));row.append(details);$('audit-list').append(row);});}
  $('proposal-form').addEventListener('submit',event=>{event.preventDefault();action($('propose-run'),async()=>{
    const evidence=[];const url=$('source-url').value.trim(),title=$('source-title').value.trim(),excerpt=$('source-excerpt').value;
    if(url||title||excerpt){if(!url||!title)throw new Error('Provide both a source title and URL, or leave the source blank.');evidence.push({url,title,excerpt});}
    const run=await api('/api/studio/runs',{method:'POST',body:JSON.stringify({query:$('run-query').value,pipeline_type:$('run-pipeline').value,evidence})});await refreshRuns();renderRun(run);say('Proposal saved. A separate reviewer must approve it before execution.');
  });});
  $('execute-run').addEventListener('click',()=>action($('execute-run'),async()=>{
    const id=selected.id;say('Executing approved local pipeline…');
    const poll=setInterval(async()=>{try{const run=await api('/api/studio/runs/'+encodeURIComponent(id));if(selected?.id===id)renderRun(run);}catch(error){say(error.message);}},750);
    try {const run=await api(`/api/studio/runs/${encodeURIComponent(id)}/execute`,{method:'POST',body:'{}'});if(selected?.id===id)renderRun(run);say('Local pipeline completed; recorded traces are ready.');}
    finally{clearInterval(poll);await refreshRuns();await refreshReport();}
  }));
  $('refresh-runs').addEventListener('click',()=>action($('refresh-runs'),refreshRuns));
  $('refresh-status').addEventListener('click',()=>action($('refresh-status'),refreshStatus));
  $('refresh-report').addEventListener('click',()=>action($('refresh-report'),refreshReport));
  $('refresh-audit').addEventListener('click',()=>action($('refresh-audit'),refreshAudit));
  const guide=[
    ['execution','1 / Frame a question. This fictional library-service example contains no account data or external provider calls.'],
    ['execution','2 / Inspect the graph. Select a node to see its dependencies. Every preview node is simulated; timings are unavailable.'],
    ['execution','3 / Follow a source. The inspector links to public documentation and marks the example as unverified.'],
    ['reviews','4 / Keep human review visible. In the signed-in workspace, a separate reviewer must approve before local execution.'],
    ['status','5 / Check the boundaries. Distinguish local execution from simulated adapters and disconnected external services.']
  ];
  function showGuide(){selectTab(guide[guideStep][0]);$('guide-instruction').textContent=guide[guideStep][1];$('guide-back').disabled=guideStep===0;$('guide-next').disabled=guideStep===guide.length-1;}
  $('guide-next').addEventListener('click',()=>{guideStep=Math.min(guide.length-1,guideStep+1);showGuide();});$('guide-back').addEventListener('click',()=>{guideStep=Math.max(0,guideStep-1);showGuide();});
  async function init(){
    if(preview){
      $('identity').textContent='Public demonstration';$('workspace-mode').textContent='Guided preview · Simulated';$('account-link').href='/auth/login';$('account-link').textContent='Sign in';$('preview-guide').hidden=false;
      ['proposal-details','refresh-runs','refresh-status','refresh-report','toolkit-link'].forEach(id=>{$(id).hidden=true;});
      const run=await api('/api/studio/preview');runs=[run];renderRun(run);$('run-list').replaceChildren(el('p',run.query,'muted'));
      $('review-list').append(el('p','Demonstration: the review step would record a reviewer decision before local execution.'));
      $('review-note').textContent='No approval is recorded or required for this public, non-sensitive fixture.';
      renderStatus({models:[{name:'Example model adapter',status:'demonstration_only',mode:'simulated'}],tools:[{name:'Preview graph',status:'fixture_only',mode:'simulated'},{name:'External inference and retrieval',status:'not_connected',mode:'live'}],scope:'Public fixture; no server health observations.',persistence:'no_preview_records_saved'});
      renderReport({scope:'Public preview has no usage samples.',proposed_runs:0,completed_runs:0,failed_runs:0,success_rate:null,average_duration_ms:null,provider_calls:0,token_usage:null,cost_usd:null,cost_note:'Preview values are not production usage or billing data.'});showGuide();return;
    }
    identity=await api('/api/studio/session');$('identity').textContent=`${identity.name} · ${identity.role}`;
    $('proposal-details').hidden=!identity.can_run;$('role-note').textContent=identity.can_run?'You can propose local runs. Execution requires a separate reviewer.':'Your role provides read access to your saved runs. Ask the workspace administrator for an operator role to propose work.';
    $('tab-audit').hidden=!identity.can_review;
    await refreshRuns();await refreshStatus();await refreshReport();if(identity.can_review)await refreshAudit();
    const mostRecent=runs[0];if(mostRecent)renderRun(await api('/api/studio/runs/'+encodeURIComponent(mostRecent.id)));
    ['pointerdown','keydown','touchstart'].forEach(type=>document.addEventListener(type,()=>{lastActivity=Date.now();},{passive:true}));
    idleTimer=setInterval(()=>{if(Date.now()-lastActivity>identity.idle_timeout_seconds*1000||Date.now()/1000>identity.expires_at){clearInterval(idleTimer);lockSession();}},10000);
  }
  init().catch(error=>say(error.message));
})();
