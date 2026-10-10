"""Bounded local text extraction. Originals are never modified."""
import base64
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from werkzeug.utils import secure_filename

MAX_FILE=5*1024*1024
MAX_TEXT=100000
OCR_PAGES=10
OCR_SECONDS=30
_slots=threading.BoundedSemaphore(1)


def revision(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _run(args,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError()
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True,timeout=remaining,
                          env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'LANG':'C.UTF-8','OMP_THREAD_LIMIT':'1'})
    return result.stdout


def ocr(content,mime,pages=None):
    if not shutil.which('tesseract') or (mime=='application/pdf' and not shutil.which('pdftoppm')):
        return '', 'unavailable'
    if pages is not None and len(pages)>OCR_PAGES:return '', 'page_limit'
    if not _slots.acquire(blocking=False):return '', 'busy'
    try:
        deadline=time.monotonic()+OCR_SECONDS
        # Private memory-backed temporary files on Linux; cleanup on every exit.
        with tempfile.TemporaryDirectory(prefix='dredge-ocr-',dir='/dev/shm' if Path('/dev/shm').is_dir() else None) as folder:
            root=Path(folder);chunks=[]
            if mime=='application/pdf':
                source=root/'source.pdf';source.write_bytes(content)
                for number in pages:
                    base=root/'page'
                    _run(['pdftoppm','-f',str(number),'-l',str(number),'-singlefile','-scale-to','3000','-png',str(source),str(base)],deadline)
                    image=base.with_suffix('.png')
                    text=_run(['tesseract',str(image),'stdout','-l','eng'],deadline).decode('utf-8')
                    chunks.append((number,text));image.unlink(missing_ok=True)
                    if sum(len(t) for _,t in chunks)>MAX_TEXT:raise ValueError()
            else:
                from PIL import Image,ImageOps
                with Image.open(io.BytesIO(content)) as source:
                    image=ImageOps.exif_transpose(source).convert('RGB');image.save(root/'page.png')
                text=_run(['tesseract',str(root/'page.png'),'stdout','-l','eng'],deadline).decode('utf-8')
                chunks=[(1,text)]
            if not any(t.strip() for _,t in chunks):return '', 'no_text'
            if sum(len(t) for _,t in chunks)>MAX_TEXT:return '', 'text_limit'
            return chunks, 'pending_review'
    except (subprocess.TimeoutExpired,TimeoutError):return '', 'timeout'
    except Exception:return '', 'failed'
    finally:_slots.release()


def read_document(file):
    name=secure_filename(file.filename or '');content=file.read(MAX_FILE+1)
    if not content or len(content)>MAX_FILE:raise ValueError('Use a file containing data, up to 5 MB.')
    text='';status='not_required';method='native';needs_ocr=False
    if name.lower().endswith('.txt'):
        text=content.decode('utf-8');mime='text/plain'
        if '\x00' in text:raise ValueError('Invalid text document.')
    elif name.lower().endswith('.pdf') and content.startswith(b'%PDF-'):
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(content))
        if reader.is_encrypted or len(reader.pages)>100:raise ValueError('Use an unencrypted PDF of up to 100 pages.')
        chunks=[];scans=[]
        for number,page in enumerate(reader.pages,1):
            chunk=page.extract_text() or '';chunks.append(chunk)
            if len(chunk.strip())<20:scans.append(number)
            if sum(map(len,chunks))>MAX_TEXT:raise ValueError('Document text is too long.')
        mime='application/pdf'
        if scans:
            needs_ocr=True;method='native_and_ocr';result,status=ocr(content,mime,scans)
            if status=='pending_review':
                for number,chunk in result:chunks[number-1]=chunk
        text='\n'.join(f'[Page {i}]\n{chunk}' for i,chunk in enumerate(chunks,1)) if needs_ocr else '\n'.join(chunks)
    elif name.lower().endswith(('.jpg','.jpeg','.png')):
        from PIL import Image
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {'JPEG','PNG'} or image.width*image.height>20000000:raise ValueError('Invalid image size or format.')
            mime='image/jpeg' if image.format=='JPEG' else 'image/png';image.verify()
        needs_ocr=True;method='ocr';result,status=ocr(content,mime)
        if status=='pending_review':text=result[0][1]
    else:raise ValueError('Use TXT, PDF, JPEG or PNG.')
    if len(text)>MAX_TEXT:raise ValueError('Document text is too long.')
    return dict(name=name,mime=mime,text=text,content=base64.b64encode(content).decode(),sha256=hashlib.sha256(content).hexdigest(),kind='original',
                extraction_method=method,ocr_required=needs_ocr,ocr_status=status,text_revision=revision(text))


def needs_review(document):
    return document.get('ocr_required',False) or document.get('mime','').startswith('image/') or (document.get('mime')=='application/pdf' and not document.get('text','').strip())


def metadata(document):
    status=document.get('ocr_status','needs_processing' if needs_review(document) else 'not_required')
    return dict(name=document['name'],ocr_status=status,extraction_method=document.get('extraction_method','legacy'))
