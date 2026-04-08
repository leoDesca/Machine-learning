from docx import Document

d=Document('SW-ML-17 Concept Paper.docx')
for i,p in enumerate(d.paragraphs):
    t=(p.text or '').strip()
    if t and (t.startswith(('I. ','II. ','III. ','IV. ','V. ','VI. ','VII. ','VIII. ','IX. ','X. ','XI. ','XII. ','XIII. ','XIV. ')) or t.startswith(('A. ','B. ','C. ','D. ','E. ')) or t.startswith('ABSTRACT') or t.startswith('INDEX TERMS')):
        print(f'[{i:03}] {t}')
