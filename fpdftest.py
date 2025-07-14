from fpdf import FPDF

title = 30
h1 = 20
h2 = 15
p = 10

left = [[0.4255136347333191, 4.228460192415935], [7.9567709058376535, 10.998810939357908]]
right = [[0.45803729406570814, 4.249140090692474]]
spaceUtilised = '26%'

pdf = FPDF(orientation="P",unit="pt",format="A4")

pdf.add_page()

pdf.set_font(family='Arial',style='B',size=title)

pdf.multi_cell(txt='Your presentation',w=0,h=50)

pdf.set_font(family='Arial',style='B',size=h1)

pdf.multi_cell(txt='Space usage',w=0,h=50)

pdf.set_font(family='Arial',style='B',size=p)

pdf.multi_cell(txt=f'Space utilised: {spaceUtilised}',w=0,h=50)

pdf.set_font(family='Arial',style='B',size=h1)

pdf.multi_cell(txt='Hand gestures',w=0,h=50)

pdf.set_font(family='Arial',style='B',size=h2)

pdf.multi_cell(txt='Graph',w=0,h=50)

pdf.image('plot.png',w=400,h=200)

pdf.set_font(family='Arial',style='B',size=h2)

pdf.multi_cell(txt='Left hand',w=0,h=50)

pdf.set_font(family='Arial',style='B',size=p)

for gesture in left:
    pdf.multi_cell(txt=f'The left hand gesture from {round(gesture[0],2)}s to {round(gesture[1],2)}s exceeded the three second recommendation',w=0,h=50)

pdf.set_font(family='Arial',style='B',size=p)

for gesture in right:
    pdf.multi_cell(txt=f'The right hand gesture from {round(gesture[0],2)}s to {round(gesture[1],2)}s exceeded the three second recommendation',w=0,h=50)

pdf.output('output.pdf')