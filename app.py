from flask import Flask, render_template, request, redirect, url_for, make_response
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from fpdf import FPDF
from shapely.geometry import LineString
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe
import re
import whisper
import evaluate
import string
import datetime
import sys

UPLOAD_FOLDER = './userVideo'
ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_duration(file_path):
    ffmpeg_path = get_ffmpeg_exe()
    
    result = subprocess.run(
        [ffmpeg_path, "-i", file_path],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError("Could not find duration in ffmpeg output.")

    hours, minutes, seconds = map(float, match.groups())
    duration = hours * 3600 + minutes * 60 + seconds
    return duration

def section_x(x_coord,sectionNo):
    for n in range(sectionNo):
        if n == 0:
            if x_coord >= 0 and x_coord <= (1280/sectionNo)*(n+1):
                return 1
        else:
            if x_coord > (1280/sectionNo)*n and x_coord <= (1280/sectionNo)*(n+1):
                return n+1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET','POST'])
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        print('Hi')
        references = [[None]]
        if 'file' not in request.files:
            return redirect(url_for('upload'))
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('upload'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if 'scriptFile' in request.files:
            print(f'request.files: {request.files}')
            scriptFile = request.files['scriptFile']
            print(scriptFile)
            if scriptFile.filename == '':
                references = [[None]]
            else:
                if scriptFile:
                    scriptFilename = secure_filename(scriptFile.filename)
                    scriptFile.save(os.path.join(app.config['UPLOAD_FOLDER'], scriptFilename))
                    scriptFile = f'userVideo/{scriptFile.filename}'
                    with open(scriptFile, 'r',encoding='utf-8') as f:
                        references = [[f.read()]]
        video = f'userVideo/{file.filename}'
        import cv2
        left_wrist_coords = []
        right_wrist_coords = []
        max_x = 0
        min_x = 100000000000000000000000
        StomachHeights = []
        kneeHeights = []
        hipHeights = []
        shoulderHeights = []
        insideBox = []
        rightSections = []
        leftSections = []
        x_positions = []
        sections = []
        sectionNo = 5
        frame_interval = 5
        actual_left_wrist_coords = []
        actual_right_wrist_coords = []
        min_frame = None
        max_frame = None

        start_time = time.time()

        model = YOLO('yolov8n-pose')

        cap = cv2.VideoCapture(video)
        frame_number = 0
        if not cap.isOpened():
            print(f"Error opening video file {video}")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                if frame_number % frame_interval == 0:
                    height, width = frame.shape[:2]
                    print(f"HEIGHTTTTT: {height}")
                    part_height = height // 9

                    # Draw 7 horizontal lines to split into 8 parts
                    #for i in range(1, 9):
                    #    y = i * part_height
                    #    print(y)
                     #   cv2.line(frame, (0, y), (width, y), (0, 255, 0), 2)
                    cv2.line(frame, (0, 406), (width, 406), (0, 255, 0), 2)
                    cv2.line(frame, (0, 563), (width, 563), (0, 255, 0), 2)
                    cv2.line(frame, (0, 138), (width, 138), (0, 255, 0), 2)
                    result = model(frame, show=True, conf=0.3, save=True)[0]
                    print(frame_number)
                    width = result.orig_shape[1]

                    #Confirm which human on the screen is a presenter
                    xyxyboxes = result.boxes.xyxy.tolist()
                    heights = []
                    for xyxybox in xyxyboxes:
                        heights.append(xyxybox[3]-xyxybox[1])
                    presenterIndex = heights.index(max(heights))
                    print(xyxyboxes[presenterIndex][0])

                    #Note the maximum and minium horizontal positions of the user
                    if xyxyboxes[presenterIndex][0] < min_x:
                        min_x = xyxyboxes[presenterIndex][0]
                        min_frame = frame_number-1
                        orig_img = result.orig_img
                        cv2.imwrite('min_frame.jpg', orig_img)
                    
                    if xyxyboxes[presenterIndex][2] > max_x:
                        max_x = xyxyboxes[presenterIndex][2]
                        max_frame = frame_number-1
                        orig_img = result.orig_img
                        cv2.imwrite('max_frame.jpg', orig_img)

                    #keypoints
                    result_keypoint_coords = result.keypoints.xyn.tolist()[presenterIndex]
                    result_keypoint_coords = result.keypoints.xyn.tolist()[0]
                    left_wrist = result_keypoint_coords[9][1]*-1
                    actual_left_wrist_coords.append((left_wrist))
                    left_wrist_coords.append((left_wrist))
                    right_wrist = result_keypoint_coords[10][1]*-1
                    actual_right_wrist_coords.append(right_wrist)
                    right_wrist_coords.append((right_wrist))
                    print(result_keypoint_coords[0][0])
                    x_positions.append(result_keypoint_coords[0][0])
                    
                    sections.append(section_x(result_keypoint_coords[0][0]*1280,sectionNo))

                    presenterStomachHeight = ((result_keypoint_coords[11][1]+result_keypoint_coords[5][1])/2+(result_keypoint_coords[12][1]+result_keypoint_coords[6][1])/2)/2*-1+ 0.1
                    
                    presenterHipHeight = (result_keypoint_coords[11][1]+result_keypoint_coords[12][1])/2*-1 + 0.1
                    print(f'Hip Height: {((result_keypoint_coords[11][1]+result_keypoint_coords[12][1])/2+0.1)*height}')
                    hipHeights.append(presenterHipHeight)
                    presenterKneeHeight = (result_keypoint_coords[13][1]+result_keypoint_coords[14][1])/2*-1 + 0.1
                    print(f'Knee Height: {((result_keypoint_coords[13][1]+result_keypoint_coords[14][1])/2+0.1)*height}')
                    kneeHeights.append(presenterKneeHeight)
                    presenterShoulderHeight = (result_keypoint_coords[5][1]+result_keypoint_coords[5][1])/2*-1
                    print(f'Shoulder height: {((result_keypoint_coords[5][1]+result_keypoint_coords[5][1])/2)*height}')
                    shoulderHeights.append(presenterShoulderHeight)
                    #rightSections.append([presenterKneeHeight,right_wrist,presenterHipHeight])
                    if right_wrist > presenterKneeHeight and right_wrist < presenterHipHeight:
                        rightSections.append(1)
                    elif right_wrist > presenterHipHeight and right_wrist < presenterShoulderHeight:
                        rightSections.append(2)
                    elif right_wrist < presenterKneeHeight:
                        rightSections.append(0)
                    else:
                        rightSections.append(3)

                    if left_wrist > presenterKneeHeight and left_wrist < presenterHipHeight:
                        leftSections.append(1)
                    elif left_wrist > presenterHipHeight and left_wrist < presenterShoulderHeight:
                        leftSections.append(2)
                    elif left_wrist < presenterKneeHeight:
                        leftSections.append(0)
                    else:
                        leftSections.append(3)
                    StomachHeights.append(presenterStomachHeight)

            except:
                left_wrist_coords.append(0)
                right_wrist_coords.append(0)
            frame_number += 1
        cap.release()
        cv2.destroyAllWindows()

        totalFrames = frame_number

        spaceUtilized = f"{int(round((max_x - min_x)/width * 100,0))}%"

        plt.clf()
        plt.plot(x_positions)
        plt.title('X-coordinate of nose of presenter per frame')
        plt.ylabel('X-position')
        plt.xlabel('Frame')
        plt.legend()
        plt.ylim(0,1)
        plt.xticks(color='w')
        plt.savefig(f"plot2.png")

        plt.clf()
        plt.plot(sections)
        plt.title('Section of presenter per frame')
        plt.ylabel('X-position')
        plt.xlabel('Frame')
        plt.legend()
        plt.xticks(color='w')
        plt.ylim(0,5)
        plt.savefig(f"plot3.png")
        
        plt.clf()
        with open('loggingFile.txt','w') as file:
            file.write(f"{right_wrist_coords}")
        plt.plot(left_wrist_coords, label='Left wrist height')
        plt.plot(right_wrist_coords, label='Right wrist height')
        plt.plot(StomachHeights, label='Stomach heights')
        print(f'stomach heights: {StomachHeights}')
        print(kneeHeights)
        plt.plot(kneeHeights, label='Knee heights')
        plt.plot(hipHeights, label='Hip heights')
        plt.plot(shoulderHeights, label='Shoulder heights')
        plt.title('Height of right and left wrist of presenter per frame (0 to 1)')
        plt.ylabel('Height (0 to 1)')
        plt.xlabel('Frame')
        plt.ylim(-1,0)
        plt.legend()
        plt.yticks(color='w')
        xmin, xmax = plt.xlim()
        plt.xticks(np.arange(0, xmax + 1, frame_interval))
        plt.savefig(f"plot.png")

        print(f'rightSections:{rightSections}')
        print(f'leftSections:{leftSections}')
        plt.clf()
        print(f"Sections of presenter's wrist")
        plt.plot(list(range(1,len(rightSections)*frame_interval,frame_interval)),rightSections,label="Right hand")
        plt.plot(list(range(1,len(leftSections)*frame_interval,frame_interval)),leftSections,label="Left hand")
        plt.legend()
        plt.title('When inside rest area')
        plt.savefig('plot2.png')

        #----------------------------------------------------------------------------------

        frame_interval = 5

        for s in range(len(rightSections)):
            if s == 0 or s == len(rightSections) - 1:
                pass
            else:
                if rightSections[s - 1] == rightSections[s + 1] and rightSections[s] != rightSections[s - 1]:
                    rightSections[s] = rightSections[s - 1]

        for s in range(len(leftSections)):
            if s == 0 or s == len(leftSections) - 1:
                pass
            else:
                if leftSections[s - 1] == leftSections[s + 1] and leftSections[s] != leftSections[s - 1]:
                    leftSections[s] = leftSections[s - 1]

        plt.plot(rightSections)
        plt.plot(leftSections)
        plt.show()

        xtime = np.arange(len(rightSections))

        section_line = LineString(np.column_stack((xtime, rightSections)))

        o_one_positive = np.column_stack((np.arange(0, len(rightSections) -1 ),np.full_like(np.arange(0, len(rightSections) -1 ), 1.05, dtype=np.float64)))
        o_one_positive = LineString(o_one_positive)
        o_one_negative = np.column_stack((np.arange(0, len(rightSections) -1 ),np.full_like(np.arange(0, len(rightSections) -1 ), 0.95, dtype=np.float64)))
        o_one_negative = LineString(o_one_negative)

        intersection_o_one_positive = section_line.intersection(o_one_positive)
        intersection_o_one_negative = section_line.intersection(o_one_negative)

        o_one_positive_intersections = []
        if intersection_o_one_positive.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_positive_intersections = sorted(((LineString(intersection_o_one_positive.geoms).xy)[0]).tolist())
        elif intersection_o_one_positive.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o')
            o_one_positive_intersections = [intersection_o_one_positive.xy[0].tolist()[0]]

        o_one_positive_intersections = sorted(o_one_positive_intersections)

        #if len(o_one_positive_intersections) % 2 != 0:
        #   o_one_positive_intersections.append(len(rightSections))

        o_one_negative_intersections = []
        if intersection_o_one_negative.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_negative_intersections = sorted(((LineString(intersection_o_one_negative.geoms).xy)[0]).tolist())
        elif intersection_o_one_negative.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o') 
            o_one_negative_intersections = [intersection_o_one_negative.xy[0].tolist()[0]]

        o_one_negative_intersections = sorted(o_one_negative_intersections)

        #if len(o_one_negative_intersections) % 2 != 0:
        #    o_one_negative_intersections.append(len(rightSections))

        for x in range(len(o_one_positive_intersections)):
            o_one_positive_intersections[x] = o_one_positive_intersections[x]*frame_interval#*10# / 3
        for x in range(len(o_one_negative_intersections)):
            o_one_negative_intersections[x] = o_one_negative_intersections[x]*frame_interval#*10# / 3

        if rightSections[0] > 1:
            o_one_positive_intersections.insert(0,0)

        if rightSections[0] < 1:
            o_one_negative_intersections.insert(0,0)

        if rightSections[-1] > 1:
            o_one_positive_intersections.append(len(rightSections)*frame_interval)

        if rightSections[-1] < 1:
            o_one_negative_intersections.append(len(rightSections)*frame_interval)

        for x in range(len(o_one_positive_intersections)):
            o_one_positive_intersections[x] = o_one_positive_intersections[x] / 30

        for x in range(len(o_one_negative_intersections)):
            o_one_negative_intersections[x] = o_one_negative_intersections[x] / 30

        print('For right hand:')
        print(o_one_positive_intersections)
        print(o_one_negative_intersections)

        #

        xtime = np.arange(len(leftSections))

        section_line = LineString(np.column_stack((xtime, leftSections)))

        o_one_positive = np.column_stack((np.arange(0, len(leftSections) -1 ),np.full_like(np.arange(0, len(leftSections) -1 ), 1.05, dtype=np.float64)))
        o_one_positive = LineString(o_one_positive)
        o_one_negative = np.column_stack((np.arange(0, len(leftSections) -1 ),np.full_like(np.arange(0, len(leftSections) -1 ), 0.95, dtype=np.float64)))
        o_one_negative = LineString(o_one_negative)

        intersection_o_one_positive = section_line.intersection(o_one_positive)
        intersection_o_one_negative = section_line.intersection(o_one_negative)

        o_one_positive_intersections = []
        if intersection_o_one_positive.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_positive_intersections = sorted(((LineString(intersection_o_one_positive.geoms).xy)[0]).tolist())
        elif intersection_o_one_positive.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o')
            o_one_positive_intersections = [intersection_o_one_positive.xy[0].tolist()[0]]

        o_one_positive_intersections = sorted(o_one_positive_intersections)

        #if len(o_one_positive_intersections) % 2 != 0:
        #   o_one_positive_intersections.append(len(leftSections))

        o_one_negative_intersections = []
        if intersection_o_one_negative.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_negative_intersections = sorted(((LineString(intersection_o_one_negative.geoms).xy)[0]).tolist())
        elif intersection_o_one_negative.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o') 
            o_one_negative_intersections = [intersection_o_one_negative.xy[0].tolist()[0]]

        o_one_negative_intersections = sorted(o_one_negative_intersections)

        #if len(o_one_negative_intersections) % 2 != 0:
        #    o_one_negative_intersections.append(len(leftSections))

        for x in range(len(o_one_positive_intersections)):
            o_one_positive_intersections[x] = o_one_positive_intersections[x]*frame_interval#*frame_interval# / 3
        for x in range(len(o_one_negative_intersections)):
            o_one_negative_intersections[x] = o_one_negative_intersections[x]*frame_interval#*frame_interval# / 3

        if leftSections[0] > 1:
            o_one_positive_intersections.insert(0,0)

        if leftSections[0] < 1:
            o_one_negative_intersections.insert(0,0)

        if leftSections[-1] > 1:
            o_one_positive_intersections.append(len(leftSections)*frame_interval)

        if leftSections[-1] < 1:
            o_one_negative_intersections.append(len(leftSections)*frame_interval)

        for x in range(len(o_one_positive_intersections)):
            o_one_positive_intersections[x] = o_one_positive_intersections[x] / 30

        for x in range(len(o_one_negative_intersections)):
            o_one_negative_intersections[x] = o_one_negative_intersections[x] / 30

        print('For left hand:')
        print(o_one_positive_intersections)
        print(o_one_negative_intersections)

        #----------------------------------------------------------------------------------

        left_wrist_coords = np.column_stack((np.arange(1, len(left_wrist_coords) + 1),left_wrist_coords))
        right_wrist_coords = np.column_stack((np.arange(1, len(right_wrist_coords) + 1),right_wrist_coords))
        StomachHeights = np.column_stack((np.arange(1, len(StomachHeights) + 1),StomachHeights))
        StomachHeights = LineString(StomachHeights)
        fps = totalFrames/(get_video_duration(rf'C:\Users\Chinmay Gogate\ProgrammingCourse\yolotest2\pose-estimation\{video}')) 
        print(f'fps: {fps}')
        threesecondframes = fps*3

        title = 25
        h1 = 20
        h2 = 15
        p = 10
        multicellHeight = 25

        pdf = FPDF(orientation="P",unit="pt",format="A4")

        pdf.add_page()

        pdf.set_font(family='Arial',style='B',size=title)

        pdf.multi_cell(txt=f'Your presentation (total frames: {totalFrames}, fps:{fps})',w=0,h=50)

        if references[0][0] != None:
            references[0][0] = references[0][0].lower()
            references[0][0] = references[0][0].translate(str.maketrans('','',string.punctuation))
            model = whisper.load_model('base.en')
            whisper.DecodingOptions(language='en', fp16=False)
            result = model.transcribe(video)
            textSpeech = result['text']
            #segmentedSpeech = []
            #segmentedResult = result['text']
            '''for segment in segmentedResult:
                if segment == '':
                    pass
                else:
                    segmentedSpeech.append(segment['text'])
                print(segmentedSpeech)'''
            #segmentedSpeech = '\n'.join(segmentedSpeech)
            textSpeech = textSpeech.lower()
            textSpeech = textSpeech.translate(str.maketrans('','',string.punctuation))
            print(textSpeech)
            ffmpeg_path = get_ffmpeg_exe()
            sacrebleu = evaluate.load('sacrebleu')
            predictions = [textSpeech]
            results = sacrebleu.compute(predictions=predictions, references=references)
            print(results)
            print(textSpeech)
            score = results['score']

            pdf.set_font(family='Arial',style='B',size=h1)

            pdf.multi_cell(txt='Speech',w=0,h=40)

            pdf.set_font(family='Arial',style='B',size=p)

            pdf.multi_cell(txt=f'{textSpeech}',w=0,h=40)

            pdf.multi_cell(txt=f'Clarity: {round(score, 2)}%',w=0,h=40)

        pdf.set_font(family='Arial',style='B',size=h1)

        pdf.multi_cell(txt='Space usage',w=0,h=40)

        pdf.set_font(family='Arial',style='B',size=p)

        pdf.multi_cell(txt=f'Space utilised: {spaceUtilized}',w=0,h=multicellHeight)

        pdf.multi_cell(txt=f'Left-most position at {datetime.timedelta(seconds=round(min_frame/fps,0))} (hours:minutes:seconds)',w=0,h=multicellHeight)

        pdf.image('min_frame.jpg',w=160,h=90)

        pdf.multi_cell(txt=f'Right-most position at {datetime.timedelta(seconds=round(max_frame/fps,0))} (hours:minutes:seconds)',w=0,h=multicellHeight)

        pdf.image('max_frame.jpg',w=160,h=90)

        pdf.multi_cell(txt='Graph of wrist',w=0,h=multicellHeight)

        pdf.image('plot2.png',w=280,h=210)

        pdf.multi_cell(txt='Graph of sections',w=0,h=multicellHeight)

        pdf.image('plot3.png',w=280,h=210)

        pdf.set_font(family='Arial',style='B',size=h1)

        pdf.multi_cell(txt='Hand gestures',w=0,h=40)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Graph of wrist positions ',w=0,h=multicellHeight)

        pdf.image('plot.png',w=280,h=210)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Left hand',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=p)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Right hand',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=p)

        response = make_response(pdf.output(dest='S').encode('latin-1'))
        response.headers.set('Content-Type', 'application/pdf')
        response.headers.set('Content-Disposition', 'inline', filename=f'presentation-report-{datetime.datetime.now().strftime("%H%M%S")}.pdf')
        print("Process finished --- %s seconds ---" % (time.time() - start_time))
        return response
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
    