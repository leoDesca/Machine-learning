from docx import Document
from pathlib import Path

def inspect(path):
    doc = Document(path)
    print('\n=== ' + Path(path).name + ' ===')
    roman = {'I.','II.','III.','IV.','V.','VI.','VII.','VIII.','IX.','X.','XI.','XII.'}
    for i,p in enumerate(doc.paragraphs):
        t=(p.text or '').strip()
        if not t:
            continue
        style = p.style.name if p.style else ''
        first = t.split()[0] if t.split() else ''
        if style.lower().startswith('heading') or first in roman or (len(t) < 90 and t.upper() == t and any(ch.isalpha() for ch in t)):
            print(f'[{i:03}] ({style}) {t[:160]}')

inspect(r'REFINED CONCEPT PAPER 6.docx')
inspect(r'SW-ML-17 Concept Paper.docx')
