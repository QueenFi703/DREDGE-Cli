import io
import subprocess
import pytest
from PIL import Image,ImageDraw,ImageFont
from werkzeug.datastructures import FileStorage
from test_studio import app,client_for
from test_casework import enable,create
from dredge.document_text import read_document


def scan(kind='PNG'):
    image=Image.new('RGB',(1400,900),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',40)
    draw.multiline_text((70,100),'FICTIONAL TRAINING DATA\nJordan Example\nJanuary amount $900\nFebruary amount $950',font=font,fill='black',spacing=25)
    buf=io.BytesIO();image.save(buf,format=kind);return buf.getvalue()


@pytest.mark.parametrize('kind,extension',[('PNG','png'),('JPEG','jpg'),('PDF','pdf')])
def test_real_local_ocr_reads_photos_and_scanned_pdf(kind,extension):
    content=scan(kind);data=read_document(FileStorage(stream=io.BytesIO(content),filename='training.'+extension))
    assert data['ocr_status']=='pending_review'
    assert 'Jordan Example' in data['text'] and '$900' in data['text'] and '$950' in data['text']
    import base64
    assert base64.b64decode(data['content'])==content


def test_review_gate_corrections_concurrency_and_access(app,monkeypatch):
    enable(app);c,h=client_for(app);case_id=create(c,h);path=f'/api/casework/cases/{case_id}/files'
    original=scan();result=c.post(path,data={'file':(io.BytesIO(original),'training.png')},headers=h)
    assert result.status_code==201 and result.get_json()['ocr_status']=='pending_review'
    file_id=result.get_json()['id'];text_path=path+'/'+file_id+'/text'
    draft=c.get(text_path).get_json();payload={'kind':'analysis','question':'Summarize','case_id':case_id,'file_ids':[file_id],'consent':True}
    calls=[];monkeypatch.setattr('dredge.casework.provider',lambda p:calls.append(p) or {'blocks':[],'usage':{}})
    assert c.post('/api/casework/ai',json=payload,headers=h).status_code==409 and not calls
    other,oh=client_for(app,'test:other')
    assert other.get(text_path).status_code==404
    assert other.post(text_path,json={'text':'leak','verified':True,'revision':draft['revision']},headers=oh).status_code==404
    assert other.post(path+'/'+file_id+'/ocr',json={},headers=oh).status_code==404
    assert c.post(text_path,json={'text':draft['text'],'verified':True,'revision':'stale'},headers=h).status_code==409
    checked=draft['text']+'\nAll pages checked against original.'
    assert c.post(text_path,json={'text':checked,'verified':False,'revision':draft['revision']},headers=h).status_code==400
    assert c.post(text_path,json={'text':checked,'verified':True,'revision':draft['revision']},headers=h).status_code==200
    assert c.post(text_path,json={'text':draft['text'],'verified':True,'revision':draft['revision']},headers=h).status_code==409
    assert c.get(path+'/'+file_id).data==original
    assert c.post('/api/casework/ai',json=payload,headers=h).status_code==200 and calls
    with app.extensions['studio_store'].connect() as db:
        assert b'Jordan Example' not in db.execute('SELECT data FROM casework_files WHERE id=?',(file_id,)).fetchone()['data']


def test_ocr_unavailable_keeps_original_and_allows_checked_transcription(app,monkeypatch):
    monkeypatch.setattr('dredge.document_text.shutil.which',lambda name:None)
    enable(app);c,h=client_for(app);case_id=create(c,h);path=f'/api/casework/cases/{case_id}/files'
    original=scan();result=c.post(path,data={'file':(io.BytesIO(original),'training.png')},headers=h).get_json();assert result['ocr_status']=='unavailable'
    endpoint=path+'/'+result['id']+'/text';draft=c.get(endpoint).get_json()
    assert c.post(endpoint,json={'text':'Jordan Example. January amount $900.','verified':True,'revision':draft['revision']},headers=h).status_code==200
    assert c.get(path+'/'+result['id']).data==original


def test_limits_and_timeout_are_explicit(monkeypatch):
    import dredge.document_text as extractor
    assert extractor.ocr(b'%PDF-', 'application/pdf',list(range(1,12)))[1]=='page_limit'
    monkeypatch.setattr(extractor,'_run',lambda *args:(_ for _ in ()).throw(subprocess.TimeoutExpired('tesseract',30)))
    result=read_document(FileStorage(stream=io.BytesIO(scan()),filename='training.png'))
    assert result['ocr_status']=='timeout' and result['text']==''
