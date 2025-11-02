from fpdf import FPDF
from flask import Flask, make_response
import datetime

# Font sizes
TITLE_SIZE = 26
H1_SIZE = 20
H2_SIZE = 16
P_SIZE = 12
LINE_HEIGHT = 25
SMALL_LINE_HEIGHT = 18
BOX_PADDING = 5

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/', methods=['GET','POST'])
@app.route('/upload', methods=['GET','POST'])
def upload():

    pdf = FPDF(orientation="P", unit="pt", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=50)

    # Title
    pdf.set_font("Arial", "B", TITLE_SIZE)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, LINE_HEIGHT, "Presentation Analysis Report", align="C")
    pdf.ln(15)


    # Section: Your Speech
    pdf.set_fill_color(220, 235, 252)  # Light blue box
    pdf.set_font("Arial", "B", H1_SIZE)
    pdf.multi_cell(0, LINE_HEIGHT, "Your Speech", fill=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(50, pdf.get_y(), 545, pdf.get_y())  # Horizontal divider
    pdf.ln(10)

    pdf.set_font("Arial", "", P_SIZE)
    pdf.multi_cell(0, LINE_HEIGHT, "HELLO! My name is Bob and I am a person!!!1!!!1!!")#, fill=True)
    #pdf.ln(10)

    # Section: Key Metrics Box (Clarity)
    pdf.set_font("Arial", "B", H2_SIZE)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, LINE_HEIGHT + BOX_PADDING, f"Clarity")#, fill=True)
    pdf.set_font("Arial", "B", P_SIZE)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT + BOX_PADDING, f"{round(19,2)}%")#, fill=True)
    pdf.set_font("Arial", "I", P_SIZE - 2)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT, f"This means of the detected words from your speech, {round(19,2)}% matched the provided script")#,fill=True)

    # Section: Script Feedback
    pdf.set_font("Arial", "B", H2_SIZE)
    pdf.multi_cell(0, LINE_HEIGHT, "Script Feedback")#, fill=True)
    pdf.set_font("Arial", "", P_SIZE)
    pdf.multi_cell(0, LINE_HEIGHT, "This is ASS")#, fill=True)  # Replace with actual feedback
    pdf.ln(10)

    # Section: Space Usage
    pdf.set_fill_color(240, 240, 240)  # Light gray for stats box
    pdf.set_font("Arial", "B", H1_SIZE)
    pdf.multi_cell(0, LINE_HEIGHT, "Space Usage", fill=True)
    pdf.line(50, pdf.get_y(), 545, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Arial", "", P_SIZE)
    stats = [
        ("Space Utilised", "-1%"),
        ("Left-most position", "00:00:01"),
        ("Right-most position", "100:00:01"),
        ("Time in corners", "4%")
    ]
    for stat, value in stats:
        pdf.multi_cell(0, SMALL_LINE_HEIGHT + BOX_PADDING, f"{stat}: {value}")#,fill=True)

    # Images with captions
    pdf.image('min_frame.jpg', w=160, h=90)
    pdf.set_font("Arial", "I", P_SIZE - 2)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT, "Figure 1: Left-most frame position")#,fill=True)

    pdf.image('max_frame.jpg', w=160, h=90)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT, "Figure 2: Right-most frame position")#,fill=True)
    pdf.ln(10)

    # Section: Hand Gestures
    pdf.set_fill_color(220, 235, 252)  # Light blue box
    pdf.set_font("Arial", "B", H1_SIZE)
    pdf.multi_cell(0, LINE_HEIGHT, "Hand Gestures", fill=True)
    pdf.line(50, pdf.get_y(), 545, pdf.get_y())
    pdf.ln(10)

    # Wrist Positions Graph
    pdf.set_font("Arial", "B", H2_SIZE)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT, "Graph of Wrist Positions")#,fill=True)
    pdf.image('plot.png', w=280, h=210)
    pdf.set_font("Arial", "I", P_SIZE - 2)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT, "Figure 3: Wrist positions over time")#,fill=True)

    pdf.set_font("Arial", "B", H2_SIZE)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT+BOX_PADDING, "Left Hand Gestures (Over Limit)")
    pdf.set_font("Arial", "", P_SIZE)
    left_gestures_over_limit = [("Wave", "00:01:05", "00:01:11")]
    for gesture, start, end in left_gestures_over_limit:
        pdf.multi_cell(0, SMALL_LINE_HEIGHT+BOX_PADDING, f"Left hand: {gesture} from {start} to {end} was too long")
    pdf.ln(5)

    pdf.set_font("Arial", "B", H2_SIZE)
    pdf.multi_cell(0, SMALL_LINE_HEIGHT+BOX_PADDING, "Right Hand Gestures (Over Limit)")
    pdf.set_font("Arial", "", P_SIZE)
    right_gestures_over_limit = [("Point", "00:02:10", "00:02:18")]
    for gesture, start, end in right_gestures_over_limit:
        pdf.multi_cell(0, SMALL_LINE_HEIGHT+BOX_PADDING, f"Right hand: {gesture} from {start} to {end} was too long")
    pdf.ln(10)

    # Footer: Timestamp
    pdf.set_y(-40)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, "C")

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    response = make_response(pdf_bytes)
    response.headers.set('Content-Type', 'application/pdf')
    response.headers.set('Content-Disposition', 'inline', filename=f'presentation-report-{datetime.datetime.now().strftime("%H%M%S")}.pdf')

    return response

if __name__ == "__main__":
    app.run(debug=True)
