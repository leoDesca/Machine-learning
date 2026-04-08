from docx import Document

d=Document('SW-ML-17 Concept Paper.docx')
for i,p in enumerate(d.paragraphs):
    t=(p.text or '').strip()
    if t and (t.isupper() or t.startswith('A.') or t.startswith('B.') or t.startswith('C.') or t.startswith('D.') or t.startswith('E.')):
        print(f'[{i:03}] {t}')
