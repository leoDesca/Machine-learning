from docx import Document

doc=Document('SW-ML-17 Concept Paper.docx')
for i,p in enumerate(doc.paragraphs[:40]):
    t=(p.text or '').strip()
    if t:
        print(f'[{i:03}] {t}')
