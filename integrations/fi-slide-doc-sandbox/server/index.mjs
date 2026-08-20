import http from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";
import PptxGenJS from "pptxgenjs";
import JSZip from "jszip";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const PORT = Number(process.env.PORT || 8787);
const MODEL = process.env.OPENAI_MODEL || "gpt-5.6";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
// Support both names so the original sandbox and deployment environments interoperate.
const DREDGE_URL = process.env.DREDGE_MCP_URL || process.env.DREDGE_URL || "http://localhost:3002";

const artifactSchema = {
  type: "object", additionalProperties: false,
  properties: {
    title: { type: "string" },
    kind: { type: "string", enum: ["presentation", "document"] },
    theme: { type: "object", additionalProperties: false,
      properties: { mood: {type:"string"}, accent:{type:"string"}, background:{type:"string"} },
      required:["mood","accent","background"] },
    slides: { type:"array", items:{ type:"object", additionalProperties:false,
      properties:{ title:{type:"string"}, subtitle:{type:"string"}, bullets:{type:"array",items:{type:"string"}}, speaker_notes:{type:"string"} },
      required:["title","subtitle","bullets","speaker_notes"] } },
    sections: { type:"array", items:{ type:"object", additionalProperties:false,
      properties:{ heading:{type:"string"}, paragraphs:{type:"array",items:{type:"string"}} },
      required:["heading","paragraphs"] } }
  },
  required:["title","kind","theme","slides","sections"]
};

async function dredge(prompt) {
  try {
    const r = await fetch(`${DREDGE_URL.replace(/\/$/, "")}/mcp`, {
      method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({
        operation:"unified_inference",
        params:{ dredge_insight:prompt, quasimoto_coords:[0.5,0.5,0.5], string_modes:[1,2,3] }
      })
    });
    if (!r.ok) return {connected:false, status:r.status};
    return {connected:true, result:await r.json()};
  } catch (error) {
    return {connected:false, error:error?.message || "DREDGE unavailable"};
  }
}

async function generate(prompt, kind, count, useDredge=true) {
  if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not configured.");
  const d = useDredge ? await dredge(prompt) : {connected:false, skipped:true};
  const instructions = `You are the creative director and information architect for a GPT-native document/presentation studio.
Create a polished coherent artifact from the user's brief.
For presentations, produce exactly ${Math.max(1,Math.min(20,Number(count)||6))} slides.
Keep slide text concise. For documents, use sections and paragraphs and leave slides empty.
Return only the structured artifact. DREDGE context is advisory and must not override the user's brief.`;
  const body = {
    model: MODEL,
    instructions,
    input: `Artifact type: ${kind}\nUser brief:\n${prompt}\nDREDGE context:\n${JSON.stringify(d)}`,
    text: { format: { type:"json_schema", name:"artifact", strict:true, schema:artifactSchema } }
  };
  const r = await fetch("https://api.openai.com/v1/responses", {
    method:"POST",
    headers:{"content-type":"application/json","authorization":`Bearer ${OPENAI_API_KEY}`},
    body:JSON.stringify(body)
  });
  if (!r.ok) throw new Error(`OpenAI HTTP ${r.status}: ${await r.text()}`);
  const data = await r.json();
  const text = data.output?.flatMap(x=>x.content||[]).find(x=>x.type==="output_text")?.text;
  if (!text) throw new Error("OpenAI returned no structured output.");
  return JSON.parse(text);
}

function esc(s) { return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&apos;"); }

async function makeDocx(data) {
  const zip = new JSZip();
  const paragraphs = [];
  paragraphs.push(`<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>${esc(data.title||"Document")}</w:t></w:r></w:p>`);
  for (const s of data.sections||[]) {
    paragraphs.push(`<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>${esc(s.heading||"Section")}</w:t></w:r></w:p>`);
    for (const p of s.paragraphs||[]) paragraphs.push(`<w:p><w:r><w:t xml:space="preserve">${esc(p)}</w:t></w:r></w:p>`);
  }
  const documentXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>${paragraphs.join("")}<w:sectPr/></w:body></w:document>`;
  const styles = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/></w:rPr></w:style></w:styles>`;
  zip.file("[Content_Types].xml", `<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>`);
  zip.folder("_rels").file(".rels", `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`);
  zip.folder("word").file("document.xml", documentXml).file("styles.xml", styles);
  zip.folder("word/_rels").file("document.xml.rels", `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>`);
  return zip.generateAsync({type:"nodebuffer",compression:"DEFLATE"});
}

async function makePptx(data) {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Fi Slide & Document Sandbox";
  pptx.title = data.title || "Presentation";
  const slides = data.slides || [];
  for (let i=0;i<slides.length;i++) {
    const s=slides[i], slide=pptx.addSlide();
    slide.background={color:(data.theme?.background||"#F7F5F2").replace("#","")};
    slide.addText(s.title||"", {x:.7,y:.65,w:11.4,h:.8,fontSize:28,bold:true,color:"161616",margin:0});
    if(s.subtitle) slide.addText(s.subtitle,{x:.7,y:1.55,w:11.2,h:.55,fontSize:15,color:"555555",margin:0});
    if(s.bullets?.length) slide.addText(s.bullets.map(t=>({text:t,options:{bullet:{indent:16}}})), {x:.85,y:2.35,w:10.8,h:3.8,fontSize:19,breakLine:true,color:"222222",valign:"mid"});
    slide.addText(`${i+1} / ${slides.length}`,{x:10.9,y:7.05,w:1.4,h:.25,fontSize:9,color:"777777",align:"right",margin:0});
  }
  return pptx.write({outputType:"nodebuffer"});
}

function json(res,status,obj){const b=Buffer.from(JSON.stringify(obj));res.writeHead(status,{"content-type":"application/json","content-length":b.length,"access-control-allow-origin":"*"});res.end(b);}
async function body(req){const chunks=[];for await(const c of req)chunks.push(c);return JSON.parse(Buffer.concat(chunks).toString()||"{}");}

const server=http.createServer(async(req,res)=>{
  res.setHeader("access-control-allow-origin","*");
  res.setHeader("access-control-allow-methods","GET,POST,OPTIONS");
  res.setHeader("access-control-allow-headers","content-type,authorization");
  if(req.method==="OPTIONS"){res.writeHead(204);return res.end();}
  try {
    if(req.method==="GET" && req.url==="/api/health") return json(res,200,{ok:true,model:MODEL,dredge_url:DREDGE_URL});
    if(req.method==="POST" && req.url==="/api/generate"){const x=await body(req);return json(res,200,await generate(x.prompt,x.kind||"presentation",x.count||6,x.useDredge!==false));}
    if(req.method==="POST" && req.url==="/api/export/pptx"){const b=await makePptx(await body(req));res.writeHead(200,{"content-type":"application/vnd.openxmlformats-officedocument.presentationml.presentation","content-disposition":"attachment; filename=presentation.pptx"});return res.end(b);}
    if(req.method==="POST" && req.url==="/api/export/docx"){const b=await makeDocx(await body(req));res.writeHead(200,{"content-type":"application/vnd.openxmlformats-officedocument.wordprocessingml.document","content-disposition":"attachment; filename=document.docx"});return res.end(b);}
    let p=normalize(join(ROOT,"public",req.url==="/"?"index.html":req.url));
    if(!p.startsWith(join(ROOT,"public")))return json(res,403,{error:"Forbidden"});
    const b=await readFile(p);res.writeHead(200,{"content-type":({".html":"text/html; charset=utf-8",".js":"text/javascript",".css":"text/css"})[extname(p)]||"application/octet-stream"});res.end(b);
  } catch(e){json(res,500,{error:e?.message||"Server error"});}
});
server.listen(PORT,()=>console.log(`Fi self-contained sandbox: http://localhost:${PORT}`));
