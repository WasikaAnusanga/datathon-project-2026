"""Export the original report content as an editable Word document."""
from pathlib import Path
from html.parser import HTMLParser
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import math

from PIL import Image, ImageDraw, ImageFont
from reportlab.platypus import Paragraph, Table, PageBreak, Spacer

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'reports/build_technical_report.py'
scope = {'__file__': str(source), '__name__': 'report_content'}
exec(compile(source.read_text(encoding='utf-8').split('\ndoc = SimpleDocTemplate(')[0], str(source), 'exec'), scope)

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
for prefix, uri in [('w', W), ('r', R), ('wp', WP), ('a', A), ('pic', PIC)]:
    ET.register_namespace(prefix, uri)

def el(parent, name, attrs=None, text=None):
    namespace, tag = name.split(':') if ':' in name else ('w', name)
    uris = {'w': W, 'r': R, 'wp': WP, 'a': A, 'pic': PIC}
    node = ET.SubElement(parent, '{%s}%s' % (uris[namespace], tag))
    for key, value in (attrs or {}).items():
        if ':' in key:
            pre, local = key.split(':')
            key = '{%s}%s' % (uris[pre], local)
        node.set(key, str(value))
    node.text = text
    return node

class RichText(HTMLParser):
    def __init__(self, paragraph, size=20, bold=False, color='142D45'):
        super().__init__()
        self.paragraph, self.size, self.bold, self.color = paragraph, size, int(bold), color
    def handle_starttag(self, tag, attrs):
        if tag in ('b', 'strong'): self.bold += 1
        if tag == 'br': el(el(self.paragraph, 'r'), 'br')
    def handle_endtag(self, tag):
        if tag in ('b', 'strong'): self.bold -= 1
    def handle_data(self, data):
        if not data: return
        run = el(self.paragraph, 'r')
        props = el(run, 'rPr')
        el(props, 'rFonts', {'w:ascii': 'Arial', 'w:hAnsi': 'Arial'})
        el(props, 'sz', {'w:val': self.size})
        el(props, 'color', {'w:val': self.color})
        if self.bold: el(props, 'b')
        node = el(run, 't', text=data)
        node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

def paragraph(parent, text, style='Copy', cell=False):
    p = el(parent, 'p')
    props = el(p, 'pPr')
    conf = {'Hero': (66, 40, 300, True), 'Section': (38, 0, 200, True),
            'Sub': (23, 90, 70, True), 'SmallCopy': (16, 0, 70, False),
            'Cell': (17, 0, 0, False), 'Copy': (19, 0, 120, False)}
    size, before, after, bold = conf.get(style, conf['Copy'])
    el(props, 'spacing', {'w:before': before, 'w:after': after, 'w:line': 270 if not cell else 240, 'w:lineRule': 'auto'})
    if style in ('Section', 'Sub', 'Hero'):
        el(props, 'keepNext')
        el(props, 'outlineLvl', {'w:val': 0 if style in ('Section', 'Hero') else 1})
    el(props, 'widowControl')
    parser = RichText(p, size, bold, '087E8B' if style == 'Sub' else '142D45')
    parser.feed(text)
    return p


class DiagramCanvas:
    """Render the original diagram drawing commands at print resolution."""
    def __init__(self):
        self.scale = 4
        self.image = Image.new('RGB', (495*4, 420*4), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.fill = 'black'
        self.stroke = 'black'
        self.linewidth = 1
        self.setFont('Body', 8.5)
    def color(self, c):
        return tuple(round(v*255) for v in (c.red, c.green, c.blue))
    def setFillColor(self, c): self.fill = self.color(c)
    def setStrokeColor(self, c): self.stroke = self.color(c)
    def setLineWidth(self, w): self.linewidth = w
    def setDash(self, *args): pass
    def setFont(self, name, size):
        self.font = ImageFont.truetype('C:/Windows/Fonts/' + ('arialbd.ttf' if name == 'BodyBold' else 'arial.ttf'), round(size*4))
    def point(self, x, y): return (x*4, (420-y)*4)
    def line(self, x1, y1, x2, y2):
        self.draw.line([self.point(x1,y1),self.point(x2,y2)], fill=self.stroke, width=max(1,round(self.linewidth*4)))
    def roundRect(self, x, y, w, h, radius, fill=1, stroke=1):
        self.draw.rounded_rectangle([self.point(x,y+h),self.point(x+w,y)], radius=radius*4,
                                    fill=self.fill if fill else None, outline=self.stroke if stroke else None, width=4)
    def drawCentredString(self, x, y, text):
        self.draw.text(self.point(x,y), text, font=self.font, fill=self.fill, anchor='ms')


doc = ET.Element('{%s}document' % W)
body = el(doc, 'body')
diagram_bytes = None
for item in scope['story']:
    if isinstance(item, Paragraph):
        paragraph(body, item.text, item.style.name)
    elif isinstance(item, PageBreak):
        el(el(el(body, 'p'), 'r'), 'br', {'w:type': 'page'})
    elif isinstance(item, Spacer):
        p = el(body, 'p')
        props = el(p, 'pPr')
        el(props, 'spacing', {'w:before': 0, 'w:after': 0, 'w:line': round(item.height*20), 'w:lineRule': 'exact'})
    elif isinstance(item, Table):
        table = el(body, 'tbl')
        props = el(table, 'tblPr')
        el(props, 'tblW', {'w:w': 9900, 'w:type': 'dxa'})
        el(props, 'tblLayout', {'w:type': 'fixed'})
        margins = el(props, 'tblCellMar')
        for edge in ('top','left','bottom','right'):
            el(margins, edge, {'w:w': 90 if edge in ('top','bottom') else 130, 'w:type': 'dxa'})
        borders = el(props, 'tblBorders')
        el(borders, 'insideH', {'w:val': 'single', 'w:sz': 3, 'w:color': 'CFDCE3'})
        grid = el(table, 'tblGrid')
        for width in item._argW: el(grid, 'gridCol', {'w:w': round(width*20)})
        for index, row in enumerate(item._cellvalues):
            tr = el(table, 'tr')
            trpr = el(tr, 'trPr')
            el(trpr, 'cantSplit')
            if index == 0: el(trpr, 'tblHeader')
            for width, value in zip(item._argW, row):
                cell = el(tr, 'tc')
                cp = el(cell, 'tcPr')
                el(cp, 'tcW', {'w:w': round(width*20), 'w:type': 'dxa'})
                if index == 0: el(cp, 'shd', {'w:fill': 'EAF3F6'})
                paragraph(cell, value.text, 'Cell', cell=True)
    elif isinstance(item, scope['Architecture']):
        canvas = DiagramCanvas()
        item.canv = canvas
        item.draw()
        stream = BytesIO()
        canvas.image.save(stream, format='PNG')
        diagram_bytes = stream.getvalue()
        canvas.image.save(ROOT / 'reports/architecture_word_preview.png')
        p = el(body, 'p')
        el(el(p, 'pPr'), 'spacing', {'w:after': 60, 'w:before': 0})
        drawing = el(el(p, 'r'), 'drawing')
        inline = el(drawing, 'wp:inline', {'distT': 0, 'distB': 0, 'distL': 0, 'distR': 0})
        cx, cy = 495*12700, 420*12700
        el(inline, 'wp:extent', {'cx':cx,'cy':cy})
        el(inline, 'wp:docPr', {'id':1,'name':'Solution architecture','descr':'Complete data input, model training, validation and prediction pipeline'})
        graphic = el(inline, 'a:graphic')
        data = el(graphic, 'a:graphicData', {'uri':PIC})
        picture = el(data, 'pic:pic')
        nv = el(picture, 'pic:nvPicPr')
        el(nv, 'pic:cNvPr', {'id':0,'name':'architecture.png'})
        el(nv, 'pic:cNvPicPr')
        fill = el(picture, 'pic:blipFill')
        el(fill, 'a:blip', {'r:embed':'rIdImage'})
        el(el(fill, 'a:stretch'), 'a:fillRect')
        sp = el(picture, 'pic:spPr')
        xfrm = el(sp, 'a:xfrm')
        el(xfrm, 'a:off', {'x':0,'y':0})
        el(xfrm, 'a:ext', {'cx':cx,'cy':cy})
        el(el(sp, 'a:prstGeom', {'prst':'rect'}), 'a:avLst')

section = el(body, 'sectPr')
el(section, 'footerReference', {'w:type':'default','r:id':'rIdFooter'})
el(section, 'pgSz', {'w:w':11906,'w:h':16838})
el(section, 'pgMar', {'w:top':900,'w:right':1000,'w:bottom':1100,'w:left':1000,'w:header':400,'w:footer':450,'w:gutter':0})
footer = ET.Element('{%s}ftr' % W)
fp = paragraph(footer, 'DATACRAFT / URBANFLOW / TECHNICAL REPORT  ·  ', 'SmallCopy')
field = el(fp, 'fldSimple', {'w:instr':'PAGE'})
el(el(field, 'r'), 't', text='1')

def xml(node): return ET.tostring(node, encoding='utf-8', xml_declaration=True)
output = ROOT / 'reports/DataCraft_UrbanFlow_Technical_Report.docx'
with ZipFile(output, 'w', ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
</Types>''')
    z.writestr('_rels/.rels', '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>''')
    z.writestr('word/_rels/document.xml.rels', '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdImage" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/architecture.png"/><Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>''')
    z.writestr('word/document.xml', xml(doc))
    z.writestr('word/footer1.xml', xml(footer))
    z.writestr('word/media/architecture.png', diagram_bytes)

with ZipFile(output) as z:
    assert z.testzip() is None
    for name in z.namelist():
        if name.endswith(('.xml','.rels')): ET.fromstring(z.read(name))
    parsed = ET.fromstring(z.read('word/document.xml'))
    assert len(parsed.findall('.//{%s}br[@{%s}type="page"]' % (W,W))) == 13
    assert len(parsed.findall('.//{%s}tbl' % W)) == sum(isinstance(item, Table) for item in scope['story'])
print(output)
