'use strict';
(() => {
  // Deliberately self-contained: no account requests, storage, uploads or AI calls.
  const scenes = [
    {title:'Welcome to the case record', pause:5, paragraphs:[
      'Welcome to DREDGE Studio. I’m Fi, the creator behind this vision at ColeWorld inc.',
      'I’m building a workspace where a caseworker can bring documents, questions and the client’s perspective into one clearer working record.',
      'Today, we’re following one fictional child-support case. There are no real client records in this demonstration. DREDGE is an independent pilot. It is not currently connected to Missouri’s MACSS or endorsed by Missouri.',
      'The question I want us to follow is simple: When two records appear to disagree, how do we understand the difference before we act on it?'
    ],cards:[['One working record','Documents → questions → client explanation → human review.'],['Independent pilot','Fictional information only. No Missouri system connection or endorsement.']]},
    {title:'Meet Jordan Example',pause:5,paragraphs:[
      'This is Jordan Example. One fictional record lists a monthly amount of nine hundred dollars. Another lists nine hundred and fifty dollars.',
      'Jordan explains that the amount changed the following month.',
      'Those numbers give us a question. They do not, by themselves, tell us that someone made a false statement or that either record is wrong. We need to check what period each record describes, what the original sources say and what still needs clarification.',
      'This is where I want DREDGE to help: making the evidence easier to follow while leaving room for the person’s explanation.'
    ],cards:[['Record A · January','$900','Fictional monthly amount. Earlier period.'],['Record B · February','$950','Fictional monthly amount. Following period.']],note:'Question: Which period and source order does each record refer to? No official balance is calculated.'},
    {title:'Bring the documents into the workspace',pause:8,paragraphs:[
      'Caseworkers receive information in many forms. A client may have a typed record, a photograph or a scanned PDF.',
      'For this demonstration, we’re using prepared fictional records. Document intake and OCR are part of the Studio’s pilot workflow, but this session is not demonstrating a new upload or extraction. Let’s follow the evidence that is available here.',
      'For supported photos and scanned PDFs, local English OCR attempts to extract text. The caseworker checks that text against the original, especially dates, names and amounts. OCR can make mistakes. Extracted text must be verified before it is used for AI review.',
      'The original document remains part of the record. We are creating a usable text layer around it.'
    ],cards:[['Prepared source text','Record A: January notice lists $900.\nRecord B: February record lists $950.'],['Verification checkpoint','Compare dates, names and amounts against the original. This prepared text is not evidence of a fresh OCR run.']],note:'No upload, extraction or submission receipt is created by this walkthrough.'},
    {title:'Inspect the evidence, then ask a better question',pause:8,paragraphs:[
      'Now we bring the records together. The question is: Compare these records, identify the difference, suggest possible explanations and prepare neutral questions to clarify it.',
      'AI discernment supports the caseworker’s review. The goal is to separate what the documents actually show from what remains uncertain.',
      'For Jordan, a useful clarification might be: Do these records cover different months, and is there a source confirming when the amount changed?',
      'A useful draft should help us ask that question and point back to the evidence. The reviewer still needs to check every date, amount and reference. AI does not decide honesty, fraud, eligibility or the outcome of this case.',
      'AI inference is not enabled for this session, so I’m illustrating the review question rather than generating a new AI response. Any example shown here is demonstration material.'
    ],cards:[['Illustrative clarification question','Do Record A and Record B cover different months? What source confirms when the amount changed?'],['Evidence to verify','Record A: January, $900.\nRecord B: February, $950.\nJordan’s explanation: the amount changed the following month.']],note:'Prepared illustration, not an AI result. No provider request is made.'},
    {title:'Make room for Jordan’s voice',pause:5,paragraphs:[
      'The client’s explanation belongs alongside the records.',
      'In the client portal, an assigned client can submit supporting documents, receive submission receipts and document hashes, and add a clarification in their own words. Recorded document activity helps them follow what happened to their submission.',
      'Jordan’s explanation is: The nine-hundred-and-fifty-dollar amount began the following month. The earlier record describes the previous month.',
      'That explanation becomes something staff can consider and verify. It should not disappear because a document comparison found a difference.',
      'We are viewing this from the staff side today. A separate verified client account and case assignment are required to demonstrate the client’s own portal session.'
    ],cards:[['Prepared client explanation','“The $950 amount began the following month. The earlier record describes the previous month.”'],['Staff-side illustration','Preserve the explanation alongside the evidence. Identity and assigned case access must be verified separately.']],note:'No client account is signed in here; no explanation or receipt is submitted.'},
    {title:'Redaction without losing the original',pause:5,paragraphs:[
      'Sometimes a working copy needs less identifying information.',
      'DREDGE supports separate redacted text copies while retaining the original record. That distinction matters: a reviewer should be able to understand which version they are viewing.',
      'This feature currently redacts text copies. It does not visually remove information from an original PDF or photograph. The copy still needs careful review before it is shared.',
      'Here we are comparing prepared fictional text examples. No new redacted copy is being created in this walkthrough.'
    ],cards:[['Original fictional text','Jordan Example: The $950 amount began the following month.'],['Prepared redacted text illustration','[REDACTED]: The $950 amount began the following month.']],note:'Original retained. A text copy is not a visually redacted PDF or photograph.'},
    {title:'Prepare the work for human review',pause:8,paragraphs:[
      'Now we can bring the relevant sources, timeline and unanswered questions into a packet for review.',
      'The goal is to give the next person a clearer starting point: what happened, what supports it, what the client explained and what still needs checking.',
      'A different authorized staff member must perform the independent approval step. Preparing a packet does not mean it has been approved, and packet approval is not the same as releasing a draft to the client.',
      'DREDGE helps organize that responsibility. Human review remains visible.',
      'This packet remains pending. We will not demonstrate a completed approval without the separate authorized reviewer.'
    ],cards:[['Prepared packet illustration','January: Record A, $900.\nFebruary: Record B, $950.\nClient explanation preserved.\nOpen question: source confirming effective date.'],['Pending independent review','Demonstration status only. No actual packet has been submitted or approved. Draft release remains a separate action.']],note:'No official balance, enforcement action or eligibility decision is made.'},
    {title:'Explore the rest of the Studio',pause:5,paragraphs:[
      'A caseworker also needs a place for the work around the case.',
      'Pages provides shared working notes with version history. Public web research can support generic policy questions without including client identifiers or attachments.',
      'DREDGE Case Law, powered by OpenAI, supports legal-research material and prepared drafts requiring qualified review. Per-case preparation requests provide scoped estimates; attorney services are not included.',
      'The intelligence workspace includes an execution graph and source-linked evidence inspector. Models and tools show capability status, and the interface distinguishes demonstrations from recorded live operations. Usage and reliability views show available recorded information; unavailable measurements should remain clearly identified.',
      'The Studio also offers expanded and compact views, plus a separate enterprise area for annual agency quotes.',
      'This graph is a simulated public demonstration, not a live agency case run.'
    ],cards:[['Pages · prepared example','Working note: Verify the period before treating two amounts as a discrepancy.\nVersion history supports reviewing changes.'],['Research and Case Law','Generic policy questions only. Qualified review required. No research or paid operation is started here.'],['Models, tools and usage','Demonstration only. No live provider observations. Token usage and billed cost are unavailable.'],['Enterprise and views','Expanded and compact workspace views. Annual agency quote requests are separate from case records.']],graph:true},
    {title:'Let the tour guide you',pause:8,paragraphs:[
      'You do not have to remember every screen from this demonstration.',
      'Select Take a tour, and the built-in orientation walks you through the workspace with pop-up guidance. It explains what each area does and how to move through it. You can go at your own pace, skip it or replay it whenever you need help.',
      'The public guided preview is available without an account. For the staff casework walkthrough, I can also guide you over Zoom using fictional training records.'
    ],cards:[['Orientation bubble · prepared illustration','Evidence inspector\nFollow source links and check provenance. A supplied source is not automatically verified.']],tour:true,note:'This is an illustration of an orientation bubble. Open the interactive preview to use the actual Take a tour controls.'},
    {title:'Close with the vision',pause:8,paragraphs:[
      'For me, the promise of DREDGE is a clearer record and a stronger human review.',
      'Help the caseworker find the evidence, understand the question and prepare the next step. Give the client a place to explain. Keep the reviewer’s responsibility visible.',
      'As you explore, I’d love to hear: Where would this save you time? What would help a parent feel heard? What would you need to verify before trusting a draft? And what should I build differently?',
      'I’m Fi. This is DREDGE Studio by ColeWorld inc., and I’m inviting you to help shape what comes next.'
    ],cards:[['A clearer record','Find the evidence. Understand the question. Preserve the explanation.'],['A stronger human review','Check the sources. Keep uncertainty visible. Let an authorized person decide the next step.']]}
  ];
  const $ = id => document.getElementById(id);
  const element = (tag,text,cls) => {const node=document.createElement(tag);node.textContent=text;if(cls)node.className=cls;return node;};
  // A relaxed reading rate plus the requested inspection pauses; no audio assumed.
  const duration = text => Math.max(5000,text.trim().split(/\s+/).length*60000/145);
  const segments=scenes.map(scene=>[...scene.paragraphs.map(duration),scene.pause*1000]);
  const total=segments.flat().reduce((a,b)=>a+b,0);
  let scene=0,part=0,elapsed=0,playing=false,timer=null,last=0,speed=1,finished=false;
  function stop(){playing=false;if(timer!==null){clearInterval(timer);timer=null;}}
  function setText(id,text){if($(id).textContent!==text)$(id).textContent=text;}
  function controls(message){setText('play',playing?'Pause':finished?'Replay demo':'Play demo');$('previous').disabled=scene===0;$('next').disabled=scene===scenes.length-1;setText('play-state',message||(playing?'Playing · captions only':finished?'Finished · replay any scene':'Paused · captions only'));const past=segments.slice(0,scene).flat().reduce((a,b)=>a+b,0)+segments[scene].slice(0,part).reduce((a,b)=>a+b,0)+elapsed;$('progress').value=Math.min(100,past/total*100);setText('duration',`About ${Math.round(total/60000/speed)} minutes at this pace`);}
  function caption(){const s=scenes[scene];$('caption').textContent=s.paragraphs[Math.min(part,s.paragraphs.length-1)];$('cue').textContent=part===s.paragraphs.length?`Pause and inspect the screen · ${s.pause} seconds`:`Narration ${part+1} of ${s.paragraphs.length}`;}
  function render(){const s=scenes[scene];$('scene-title').textContent=s.title;$('scene-number').textContent=`Scene ${scene+1} of ${scenes.length}`;document.querySelectorAll('#scene-list button').forEach((b,i)=>{if(i===scene)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});const cards=element('div','','evidence-cards');for(const c of s.cards){const card=element('article','','evidence-card');card.append(element('h3',c[0]));if(c.length===3){card.append(element('strong',c[1],'amount'),element('p',c[2]));}else card.append(element('p',c[1]));if(s.tour)card.classList.add('tour-example');cards.append(card);}$('stage').replaceChildren(cards);if(s.graph){const graph=element('div','','graph-steps');graph.setAttribute('aria-label','Simulated graph');for(const label of ['Frame →','Sources →','Clarify →','Human review'])graph.append(element('span',label));$('stage').append(element('p','Simulated public graph','stage-note'),graph);}if(s.note)$('stage').append(element('p',s.note,'stage-note'));if(s.tour){const link=element('a','Open the interactive preview and select Take a tour ↗');link.href='/preview';$('stage').append(link);}caption();controls();}
  function navigate(index){stop();scene=Math.max(0,Math.min(scenes.length-1,index));part=0;elapsed=0;finished=false;render();$('scene-title').focus();}
  function tick(){const now=performance.now();elapsed+=(now-last)*speed;last=now;while(playing&&elapsed>=segments[scene][part]){elapsed-=segments[scene][part];part++;if(part>=segments[scene].length){if(scene===scenes.length-1){elapsed=segments[scene][segments[scene].length-1];part=segments[scene].length-1;finished=true;stop();controls();$('progress').value=100;return;}scene++;part=0;render();}else caption();}controls();}
  function begin(focus=false){if(playing)return;if(finished)navigate(0);playing=true;last=performance.now();timer=setInterval(tick,250);controls();if(focus)$('scene-title').focus();}
  $('play').addEventListener('click',()=>{if(playing){tick();stop();controls();return;}begin(true);});
  $('previous').addEventListener('click',()=>navigate(scene-1));$('next').addEventListener('click',()=>navigate(scene+1));$('restart').addEventListener('click',()=>{navigate(0);begin(true);});
  $('pace').addEventListener('change',()=>{if(playing)tick();stop();const value=Number($('pace').value);speed=[0.8,1,1.25].includes(value)?value:1;controls();});
  $('scene-list').replaceChildren();
  scenes.forEach((s,i)=>{const button=element('button',`${String(i+1).padStart(2,'0')}  ${s.title}`);button.type='button';button.addEventListener('click',()=>navigate(i));$('scene-list').append(button);const section=element('section','');section.append(element('h3',`${i+1}. ${s.title}`));for(const p of s.paragraphs)section.append(element('p',p));section.append(element('p',`[Pause ${s.pause} seconds.]`,'muted'));$('transcript').append(section);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){stop();controls('Paused while this tab is hidden. Select Play demo to continue.');}});
  window.addEventListener('pagehide',()=>{stop();controls();});
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&playing){stop();controls();}});
  ['play','previous','next','restart','pace'].forEach(id=>{$(id).disabled=false;});
  render();$('demo-readiness').hidden=true;
  if(document.hidden)controls('Paused while this tab is hidden. Select Play demo to continue.');else begin();
  document.querySelector('.transcript').addEventListener('toggle',()=>{if(document.querySelector('.transcript').open&&playing){stop();controls('Paused for the full narration. Select Play demo to continue.');}});
})();
