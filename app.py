from flask import Flask, render_template, request, redirect, url_for, make_response
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET','POST'])
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
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
        aboveStomachlevel = []
        addLeftIntersectionAtBeginning = False
        addRightIntersectionAtBeginning = False

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
                if frame_number % 2 == 0:
                    result = model(frame, show=False, conf=0.3, save=True)[0]
                    print(frame_number)
                    width = result.orig_shape[1]
                    xyxyboxes = result.boxes.xyxy.tolist()

                    heights = []

                    for xyxybox in xyxyboxes:
                        heights.append(xyxybox[3]-xyxybox[1])

                    presenterIndex = heights.index(max(heights))

                    print(xyxyboxes[presenterIndex][0])

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

                    result_keypoint_coords = result.keypoints.xyn.tolist()[presenterIndex]

                    result_keypoint_coords = result.keypoints.xyn.tolist()[0]
                    left_wrist = result_keypoint_coords[9][1]
                    left_wrist_coords.append((left_wrist*-1))
                    right_wrist = result_keypoint_coords[10][1]
                    right_wrist_coords.append((right_wrist*-1))

                    presenterStomachHeight = ((result_keypoint_coords[11][1]+result_keypoint_coords[5][1])/2+(result_keypoint_coords[12][1]+result_keypoint_coords[6][1])/2)/2

                    StomachHeights.append(presenterStomachHeight*-1)

                    if left_wrist < presenterStomachHeight or right_wrist < presenterStomachHeight:
                        if frame_number-2 == 0:
                            if left_wrist < presenterStomachHeight:
                                addLeftIntersectionAtBeginning = True
                            if right_wrist < presenterStomachHeight:
                                addRightIntersectionAtBeginning = True
                        aboveStomachlevel.append(1)
                    else:
                        aboveStomachlevel.append(0)
            except:
                left_wrist_coords.append(0)
                right_wrist_coords.append(0)
            frame_number += 1
        cap.release()
        cv2.destroyAllWindows()

        totalFrames = frame_number

        spaceUtilized = f"{int(round((max_x - min_x)/width * 100,0))}%"

        plt.plot(left_wrist_coords, label='Left wrist height')
        plt.plot(right_wrist_coords, label='Right wrist height')
        plt.plot(StomachHeights, label='Stomach heights')
        plt.title('Height of right and left wrist of presenter per frame (0 to 1)')
        plt.ylabel('Height (0 to 1)')
        plt.xlabel('Frame')
        plt.ylim(-1,0)
        plt.legend()
        plt.yticks(color='w')
        xmin, xmax = plt.xlim()
        plt.xticks(np.arange(0, xmax + 1, 5))
        plt.savefig(f"plot.png")

        left_wrist_coords = np.column_stack((np.arange(1, len(left_wrist_coords) + 1),left_wrist_coords))
        right_wrist_coords = np.column_stack((np.arange(1, len(right_wrist_coords) + 1),right_wrist_coords))
        StomachHeights = np.column_stack((np.arange(1, len(StomachHeights) + 1),StomachHeights))
        StomachHeights = LineString(StomachHeights)
        fps = totalFrames/(get_video_duration(rf'C:\Users\Chinmay Gogate\ProgrammingCourse\yolotest2\pose-estimation\{video}')) 
        print(f'fps: {fps}')
        threesecondframes = fps*3

        first_line_left = LineString(left_wrist_coords)
        intersection_left = first_line_left.intersection(StomachHeights)

        first_line_right = LineString(right_wrist_coords)
        intersection_right = first_line_right.intersection(StomachHeights)

        print(intersection_left)

        print(intersection_right)

        left_stomach_intersections = []
        right_stomach_intersections = []

        if intersection_left.geom_type == 'MultiPoint':
            plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            left_stomach_intersections = sorted(((LineString(intersection_left.geoms).xy)[0]).tolist())
        elif intersection_left.geom_type == 'Point':
            plt.plot(*intersection_left.xy, 'o')
            left_stomach_intersections = [intersection_left.xy[0].tolist()[0]]

        if intersection_right.geom_type == 'MultiPoint':
            plt.plot(*LineString(intersection_right.geoms).xy, 'o')
            right_stomach_intersections = sorted(((LineString(intersection_right.geoms).xy)[0]).tolist())
        elif intersection_right.geom_type == 'Point':
            plt.plot(*intersection_right.xy, 'o')
            right_stomach_intersections = [intersection_right.xy[0].tolist()[0]]

        if addLeftIntersectionAtBeginning == True:
            left_stomach_intersections.append(0)

        if len(left_stomach_intersections) % 2 != 0:
            left_stomach_intersections.append(totalFrames)

        if addRightIntersectionAtBeginning == True:
            right_stomach_intersections.append(0)

        if len(right_stomach_intersections) % 2 != 0:
            right_stomach_intersections.append(totalFrames)

        left_stomach_intersections = sorted(left_stomach_intersections)
        right_stomach_intersections = sorted(right_stomach_intersections)

        print(left_stomach_intersections)
        print(right_stomach_intersections)

        left_hand_gestures_over_limit = []
        right_hand_gestures_over_limit = []

        for intersection in left_stomach_intersections:
            if left_stomach_intersections.index(intersection) % 2 == 0:
                gesture_start = intersection
                print(f'gesture start: {gesture_start}')
                gesture_end = left_stomach_intersections[left_stomach_intersections.index(intersection)+1]
                print(f'gesture end: {gesture_end}')
                gesture_time = gesture_end - gesture_start
                print(gesture_time)
                if gesture_time >= threesecondframes:
                    left_hand_gestures_over_limit.append([gesture_start/fps,gesture_end/fps])

        for intersection in right_stomach_intersections:
            if right_stomach_intersections.index(intersection) % 2 == 0:
                gesture_start = intersection
                print(f'gesture start: {gesture_start}')
                gesture_end = right_stomach_intersections[right_stomach_intersections.index(intersection)+1]
                print(f'gesture end: {gesture_end}')
                gesture_time = gesture_end - gesture_start
                print(gesture_time)
                if gesture_time >= threesecondframes:
                    right_hand_gestures_over_limit.append([gesture_start/fps,gesture_end/fps])

        print(f'left hand: {left_hand_gestures_over_limit}')
        print(f'right hand: {right_hand_gestures_over_limit}')

        title = 25
        h1 = 20
        h2 = 15
        p = 10
        multicellHeight = 25

        pdf = FPDF(orientation="P",unit="pt",format="A4")

        pdf.add_page()

        pdf.set_font(family='Arial',style='B',size=title)

        pdf.multi_cell(txt='Your presentation',w=0,h=50)

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

        pdf.multi_cell(txt=f'Left-most position (at {round(min_frame/16.82,2)}s):',w=0,h=multicellHeight)

        pdf.image('min_frame.jpg',w=160,h=90)

        pdf.multi_cell(txt=f'Right-most position (at {round(max_frame/16.82,2)}s):',w=0,h=multicellHeight)

        pdf.image('max_frame.jpg',w=160,h=90)

        pdf.set_font(family='Arial',style='B',size=h1)

        pdf.multi_cell(txt='Hand gestures',w=0,h=40)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Graph',w=0,h=multicellHeight)

        pdf.image('plot.png',w=400,h=300)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Left hand',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=p)

        if len(left_hand_gestures_over_limit) > 0:
            for gesture in left_hand_gestures_over_limit:
                pdf.multi_cell(txt=f'The left hand gesture from {round(gesture[0],2)}s to {round(gesture[1],2)}s exceeded the three second recommendation',w=0,h=multicellHeight)
        else:
            pdf.multi_cell(txt=f'No right hand gestures exceeded the three second recommendation',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Right hand',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=p)

        if len(right_hand_gestures_over_limit) > 0:
            for gesture in right_hand_gestures_over_limit:
                pdf.multi_cell(txt=f'The right hand gesture from {round(gesture[0],2)}s to {round(gesture[1],2)}s exceeded the three second recommendation',w=0,h=multicellHeight)
        else:
            pdf.multi_cell(txt=f'No right hand gestures exceeded the three second recommendation',w=0,h=multicellHeight)

        response = make_response(pdf.output(dest='S').encode('latin-1'))
        response.headers.set('Content-Type', 'application/pdf')
        response.headers.set('Content-Disposition', 'attachment', filename=f'presentation-report-{datetime.datetime.now().strftime("%H%M%S")}.pdf')
        print("Process finished --- %s seconds ---" % (time.time() - start_time))
        return response
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
    