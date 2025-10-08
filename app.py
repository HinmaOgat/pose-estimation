from flask import Flask, render_template, request, redirect, url_for, make_response
print(1)
from werkzeug.utils import secure_filename
print(2)
import os
print(3)
from ultralytics import YOLO
print(4)
import matplotlib
print(5)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(6)
import numpy as np
print(7)
import time
print(8)
import cv2
print(9)
from fpdf import FPDF
print(10)
from shapely.geometry import LineString
print(11)
from imageio_ffmpeg import get_ffmpeg_exe
print(12)
import subprocess
print(13)
from imageio_ffmpeg import get_ffmpeg_exe
print(14)
import re
print(15)

import string
print(18)
import datetime
print(19)
import sys
print(20)
from unidecode import unidecode

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

#Flask stuff

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
            #If the user has submitted a script file, it means they want their speech analysed. 
            scriptFile = request.files['scriptFile']
            if scriptFile.filename == '':
                references = [[None]]
            else:
                if scriptFile:
                    scriptFilename = secure_filename(scriptFile.filename)
                    scriptFile.save(os.path.join(app.config['UPLOAD_FOLDER'], scriptFilename))
                    scriptFile = f'userVideo/{scriptFile.filename}'
                    with open(scriptFile, 'r',encoding='utf-8') as f:
                        references = [[f.read()]]
                        #references being the script file
        video = f'userVideo/{file.filename}'
        import cv2
        left_wrist_coords = []
        right_wrist_coords = []
        max_x = 0
        min_x = 100000000000000000000000
        StomachHeights = []
        rightSections = []
        leftSections = []
        x_positions = []
        sections = []
        sectionNo = 5
        frame_interval = 5
        min_frame = None
        max_frame = None

        model = YOLO('yolov8n-pose')

        start_time = time.time()

        cap = cv2.VideoCapture(video)
        frame_number = 0
        if not cap.isOpened():
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                if frame_number % frame_interval == 0:
                    height, width = frame.shape[:2]
                    #print(f"HEIGHTTTTT: {height}")
                    part_height = height // 9
                    if frame_number != 0:
                        pass
                        cv2.line(frame, (0, round(rest_Top*720*-1)), (width, round(rest_Top*720*-1)), (0, 255, 0), 2)
                        cv2.line(frame, (0, round(rest_Bottom*720*-1)), (width, round(rest_Bottom*720*-1)), (0, 255, 0), 2)

                    result = model(frame, show=False, conf=0.3, save=True)[0]
                    width = result.orig_shape[1]

                    #Confirm which human on the screen is a presenter (whichever has the greatest height)
                    xyxyboxes = result.boxes.xyxy.tolist()
                    heights = []
                    for xyxybox in xyxyboxes:
                        heights.append(xyxybox[3]-xyxybox[1])
                    presenterIndex = heights.index(max(heights))

                    #Note the maximum and minimum horizontal positions of the user. If the user is at the leftest or rightest point that they have been of the frames analysed, that is marked as the new min and max x positions
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

                    #Body keypoints
                    result_keypoint_coords = result.keypoints.xyn.tolist()[presenterIndex]
                    result_keypoint_coords = result.keypoints.xyn.tolist()[0]
                    left_wrist = result_keypoint_coords[9][1]*-1
                    left_wrist_coords.append((left_wrist))
                    right_wrist = result_keypoint_coords[10][1]*-1
                    right_wrist_coords.append((right_wrist))
                    x_positions.append(result_keypoint_coords[0][0])
                    
                    if frame_number == 0:
                        restArea = (left_wrist + right_wrist) / 2
                        rest_Top = restArea + 0.1
                        rest_Bottom = restArea - 0.1

                    #Adding the horizontal section the user is in
                    sections.append(section_x(result_keypoint_coords[0][0]*1280,sectionNo))

                    #The following code is to get the stomach, knee, hip, shoulder heights of the presenter. It is done by averaging the y-coordinate of the user's left and right shoulder, hip, etc

                    #presenterStomachHeight = ((result_keypoint_coords[11][1]+result_keypoint_coords[5][1])/2+(result_keypoint_coords[12][1]+result_keypoint_coords[6][1])/2)/2*-1+ 0.1
                    
                    #presenterHipHeight = (result_keypoint_coords[11][1]+result_keypoint_coords[12][1])/2*-1 + 0.1
                    #print(f'Hip Height: {((result_keypoint_coords[11][1]+result_keypoint_coords[12][1])/2+0.1)*height}')
                    #hipHeights.append(presenterHipHeight)
                    #presenterKneeHeight = (result_keypoint_coords[13][1]+result_keypoint_coords[14][1])/2*-1 + 0.1
                    #print(f'Knee Height: {((result_keypoint_coords[13][1]+result_keypoint_coords[14][1])/2+0.1)*height}')
                    #kneeHeights.append(presenterKneeHeight)
                    #presenterShoulderHeight = (result_keypoint_coords[5][1]+result_keypoint_coords[5][1])/2*-1
                    #print(f'Shoulder height: {((result_keypoint_coords[5][1]+result_keypoint_coords[5][1])/2)*height}')
                    #shoulderHeights.append(presenterShoulderHeight)

                    #For the right wrist, determining which 'section' of the user's body it is in

                    if right_wrist > rest_Bottom and right_wrist < rest_Top:
                        rightSections.append(1)
                    elif right_wrist > rest_Top:
                        rightSections.append(2)
                    elif right_wrist < rest_Bottom:
                        rightSections.append(0)

                    #Same for the left wrist

                    if left_wrist > rest_Bottom and left_wrist < rest_Top:
                        leftSections.append(1)
                    elif left_wrist > rest_Top:
                        leftSections.append(2)
                    elif left_wrist < rest_Bottom:
                        leftSections.append(0)
                    '''
                    if right_wrist > presenterKneeHeight and right_wrist < presenterHipHeight:
                        rightSections.append(0)
                    elif right_wrist > presenterHipHeight and right_wrist < presenterShoulderHeight:
                        rightSections.append(1)
                    elif right_wrist < presenterKneeHeight:
                        rightSections.append(-1)
                    else:
                        rightSections.append(2)

                    #Same for the left wrist
                    if left_wrist > presenterKneeHeight and left_wrist < presenterHipHeight:
                        leftSections.append(0)
                    elif left_wrist > presenterHipHeight and left_wrist < presenterShoulderHeight:
                        leftSections.append(1)
                    elif left_wrist < presenterKneeHeight:
                        leftSections.append(-1)
                    else:
                        leftSections.append(2)
                    StomachHeights.append(presenterStomachHeight)'''

            except:
                left_wrist_coords.append(0)
                right_wrist_coords.append(0)
            frame_number += 1
        cap.release()
        cv2.destroyAllWindows() #Frames done analysing!

        totalFrames = frame_number

        spaceUtilized = f"{int(round((max_x - min_x)/width * 100,0))}%"

        #Plotting the presenter's horizontal position...
        plt.clf()
        plt.plot(x_positions,label='x positions')
        plt.title('X-coordinate of nose of presenter per frame')
        plt.ylabel('X-position')
        plt.xlabel('Frame')
        plt.legend()
        plt.ylim(0,1)
        plt.xticks(color='w')
        plt.savefig(f"plot2.png")

        #and their horizontal sections...
        plt.clf()
        plt.plot(sections,label='sections')
        plt.title('Section of presenter per frame')
        plt.ylabel('X-position')
        plt.xlabel('Frame')
        plt.legend()
        plt.xticks(color='w')
        plt.ylim(0,5)
        plt.savefig(f"plot3.png")
        
        #And their wrist coordinates & hip, shoulder, knee, stomach heights...
        plt.clf()
        with open('loggingFile.txt','w') as file:
            file.write(f"{right_wrist_coords}")
        plt.plot(left_wrist_coords, label='Left wrist height')
        plt.plot(right_wrist_coords, label='Right wrist height')
        #plt.hlines(y=[-0.436, -0.341], xmin=0, xmax=len(left_wrist_coords), colors=['k', 'k'], linestyles=['-', '--', ':'])
        #plt.plot(StomachHeights, label='Stomach heights')
        #plt.plot(kneeHeights, label='Knee heights')
        #plt.plot(hipHeights, label='Hip heights')
        #plt.plot(shoulderHeights, label='Shoulder heights')
        plt.title('Height of right and left wrist of presenter per frame (0 to 1)')
        plt.ylabel('Height (0 to 1)')
        plt.xlabel('Frame')
        plt.ylim(-1,0)
        plt.legend()
        xmin, xmax = plt.xlim()
        plt.xticks(np.arange(0, xmax + 1, frame_interval))
        plt.savefig(f"plot.png")

        #and their wrist sections!
        plt.clf()
        plt.plot(list(range(1,len(rightSections)*frame_interval,frame_interval)),rightSections,label="Right hand")
        plt.plot(list(range(1,len(leftSections)*frame_interval,frame_interval)),leftSections,label="Left hand")
        plt.legend()
        plt.title('When inside rest area')
        plt.savefig('plot2.png')

        #----------------------------------------------------------------------------------
        
        #These loops are to eliminate any fluctuations; if the user's hand goes from section 2 to section one for one time and then back to section 2, it was probably just due to them being in that approximate area instead of a hand gesture. This eliminates this; if the data is 2,2,1,2 (where 1 is clearly just a fluctuation) it changes it to 2,2,2,2
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

        #The following code gets the intersection between the user's right wrist section graph and the y-values 1.05 and 0.95. As the user's wrist is at y=1 when in their 'rest area', this detects when they leave that area (go to 0 or 2 or 3), thus indicating the start of the gesture

        xtime = np.arange(len(rightSections))

        section_line = LineString(np.column_stack((xtime, rightSections)))

        o_one_positive = np.column_stack((np.arange(0, len(rightSections) -1 ),np.full_like(np.arange(0, len(rightSections) -1 ), 1.05, dtype=np.float64)))
        o_one_positive = LineString(o_one_positive)
        o_one_negative = np.column_stack((np.arange(0, len(rightSections) -1 ),np.full_like(np.arange(0, len(rightSections) -1 ), 0.95, dtype=np.float64)))
        o_one_negative = LineString(o_one_negative)

        intersection_o_one_positive = section_line.intersection(o_one_positive)
        intersection_o_one_negative = section_line.intersection(o_one_negative)

        o_one_positive_intersections_right = []
        if intersection_o_one_positive.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_positive_intersections_right = sorted(((LineString(intersection_o_one_positive.geoms).xy)[0]).tolist())
        elif intersection_o_one_positive.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o')
            o_one_positive_intersections_right = [intersection_o_one_positive.xy[0].tolist()[0]]

        o_one_positive_intersections_right = sorted(o_one_positive_intersections_right)

        #if len(o_one_positive_intersections) % 2 != 0:
        #   o_one_positive_intersections.append(len(rightSections))

        o_one_negative_intersections_right = []
        if intersection_o_one_negative.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_negative_intersections_right = sorted(((LineString(intersection_o_one_negative.geoms).xy)[0]).tolist())
        elif intersection_o_one_negative.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o') 
            o_one_negative_intersections_right = [intersection_o_one_negative.xy[0].tolist()[0]]

        o_one_negative_intersections_right = sorted(o_one_negative_intersections_right)

        #if len(o_one_negative_intersections) % 2 != 0:
        #    o_one_negative_intersections.append(len(rightSections))

        for x in range(len(o_one_positive_intersections_right)):
            o_one_positive_intersections_right[x] = o_one_positive_intersections_right[x]*frame_interval#*10# / 3
        for x in range(len(o_one_negative_intersections_right)):
            o_one_negative_intersections_right[x] = o_one_negative_intersections_right[x]*frame_interval#*10# / 3

        if rightSections[0] > 1:
            o_one_positive_intersections_right.insert(0,0)

        if rightSections[0] < 1:
            o_one_negative_intersections_right.insert(0,0)

        if rightSections[-1] > 1:
            o_one_positive_intersections_right.append(len(rightSections)*frame_interval)

        if rightSections[-1] < 1:
            o_one_negative_intersections_right.append(len(rightSections)*frame_interval)

        for x in range(len(o_one_positive_intersections_right)):
            o_one_positive_intersections_right[x] = o_one_positive_intersections_right[x] / 30

        for x in range(len(o_one_negative_intersections_right)):
            o_one_negative_intersections_right[x] = o_one_negative_intersections_right[x] / 30

        #print('For right hand:')
        #print(o_one_positive_intersections_right)
        #print(o_one_negative_intersections_right)

        #-----------------------------------------------------------

        #And same for the left wrist

        xtime = np.arange(len(leftSections))

        section_line = LineString(np.column_stack((xtime, leftSections)))

        o_one_positive = np.column_stack((np.arange(0, len(leftSections) -1 ),np.full_like(np.arange(0, len(leftSections) -1 ), 1.05, dtype=np.float64)))
        o_one_positive = LineString(o_one_positive)
        o_one_negative = np.column_stack((np.arange(0, len(leftSections) -1 ),np.full_like(np.arange(0, len(leftSections) -1 ), 0.95, dtype=np.float64)))
        o_one_negative = LineString(o_one_negative)

        intersection_o_one_positive = section_line.intersection(o_one_positive)
        intersection_o_one_negative = section_line.intersection(o_one_negative)

        o_one_positive_intersections_left = []
        if intersection_o_one_positive.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_positive_intersections_left = sorted(((LineString(intersection_o_one_positive.geoms).xy)[0]).tolist())
        elif intersection_o_one_positive.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o')
            o_one_positive_intersections_left = [intersection_o_one_positive.xy[0].tolist()[0]]

        o_one_positive_intersections_left = sorted(o_one_positive_intersections_left)

        #if len(o_one_positive_intersections) % 2 != 0:
        #   o_one_positive_intersections.append(len(leftSections))

        o_one_negative_intersections_left = []
        if intersection_o_one_negative.geom_type == 'MultiPoint':
            #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
            o_one_negative_intersections_left = sorted(((LineString(intersection_o_one_negative.geoms).xy)[0]).tolist())
        elif intersection_o_one_negative.geom_type == 'Point':
            #plt.plot(*intersection_left.xy, 'o') 
            o_one_negative_intersections_left = [intersection_o_one_negative.xy[0].tolist()[0]]

        o_one_negative_intersections_left = sorted(o_one_negative_intersections_left)

        #if len(o_one_negative_intersections) % 2 != 0:
        #    o_one_negative_intersections.append(len(leftSections))

        for x in range(len(o_one_positive_intersections_left)):
            o_one_positive_intersections_left[x] = o_one_positive_intersections_left[x]*frame_interval#*frame_interval# / 3
        for x in range(len(o_one_negative_intersections_left)):
            o_one_negative_intersections_left[x] = o_one_negative_intersections_left[x]*frame_interval#*frame_interval# / 3

        if leftSections[0] > 1:
            o_one_positive_intersections_left.insert(0,0)

        if leftSections[0] < 1:
            o_one_negative_intersections_left.insert(0,0)

        if leftSections[-1] > 1:
            o_one_positive_intersections_left.append(len(leftSections)*frame_interval)

        if leftSections[-1] < 1:
            o_one_negative_intersections_left.append(len(leftSections)*frame_interval)

        for x in range(len(o_one_positive_intersections_left)):
            o_one_positive_intersections_left[x] = o_one_positive_intersections_left[x] / 30

        for x in range(len(o_one_negative_intersections_left)):
            o_one_negative_intersections_left[x] = o_one_negative_intersections_left[x] / 30

        #print('For left hand:')
        #print(o_one_positive_intersections_left)
        #print(o_one_negative_intersections_left)

        #----------------------------------------------------------------------------------

        #This was old code which I am keeping in case I need it, but it originally was part of code to detect interactions between the user's wrists and their stomach
        #left_wrist_coords = np.column_stack((np.arange(1, len(left_wrist_coords) + 1),left_wrist_coords))
        #right_wrist_coords = np.column_stack((np.arange(1, len(right_wrist_coords) + 1),right_wrist_coords))
        StomachHeights = np.column_stack((np.arange(1, len(StomachHeights) + 1),StomachHeights))
        StomachHeights = LineString(StomachHeights)

        #Gets the fps
        fps = totalFrames/(get_video_duration(rf'C:\Users\Chinmay Gogate\ProgrammingCourse\yolotest2\pose-estimation\{video}')) 
        threesecondframes = fps*3

        #The following code is for the generation of the PDF file

        title = 25
        h1 = 20
        h2 = 15
        p = 10
        multicellHeight = 25

        pdf = FPDF(orientation="P",unit="pt",format="A4")

        pdf.add_page()

        pdf.set_font(family='Arial',style='B',size=title)

        pdf.multi_cell(txt=f'Your presentation (total frames: {totalFrames}, fps:{fps})',w=0,h=50)

        #Earlier, references was the script file
        
        if references[0][0] != None:
            import whisper
            print(16)
            import evaluate
            print(17)
            references[0][0] = references[0][0].lower()
            references[0][0] = references[0][0].translate(str.maketrans('','',string.punctuation)).replace("’",'').replace("—",'').replace("\n",'')
            #loads the whisper model
            model = whisper.load_model('base.en')
            whisper.DecodingOptions(language='en', fp16=False)
            result = model.transcribe(video)
            textSpeech = result['text']
            #segmentedSpeech = '\n'.join(segmentedSpeech)
            textSpeech = textSpeech.lower()
            textSpeech = textSpeech.translate(str.maketrans('','',string.punctuation)).replace("’",'').replace("—",'').replace("\n",'')
            ffmpeg_path = get_ffmpeg_exe()
            #sacrebleu is a tool used for setting 'scores' for how well two texts match each other. It is actually for assessing the quality of machine-generated text, which is similar to what I am trying to see (how well the speech-to-text understands the user's speaking, which will help me gauge how clear they are speaking)
            sacrebleu = evaluate.load('sacrebleu')
            predictions = [textSpeech]
            print('------------------------------------------------')
            print('PREDICTION')
            print(predictions)
            print('--------')
            print('REFERENCES')
            print(references)
            print('------------------------------------------------')
            results = sacrebleu.compute(predictions=predictions, references=references)
            #Gets the clarity percentage
            score = results['score']

            pdf.set_font(family='Arial',style='B',size=h1)

            pdf.multi_cell(txt='Prediction',w=0,h=40)

            predictions[0] = unidecode(predictions[0])

            pdf.multi_cell(txt=f'{str(predictions)}',w=0,h=40)

            references[0][0] = unidecode(references[0][0])

            pdf.multi_cell(txt='References',w=0,h=40)

            pdf.multi_cell(txt=f'{str(references)}',w=0,h=40)

            pdf.set_font(family='Arial',style='B',size=p)

            pdf.multi_cell(txt=f'{textSpeech}',w=0,h=40)

            pdf.multi_cell(txt=f'Clarity: {round(score, 2)}%',w=0,h=40)

            #Remember, all that was just if the user had submitted a script. If not, it is not shown and only the results of their body analysis are returned

        pdf.set_font(family='Arial',style='B',size=h1)

        pdf.multi_cell(txt='Space usage',w=0,h=40)

        pdf.set_font(family='Arial',style='B',size=p)

        pdf.multi_cell(txt=f'Space utilised: {spaceUtilized}',w=0,h=multicellHeight)

        pdf.multi_cell(txt=f'Left-most position at {datetime.timedelta(seconds=round(min_frame/fps,0))} (hours:minutes:seconds)',w=0,h=multicellHeight)

        pdf.image('min_frame.jpg',w=160,h=90)

        pdf.multi_cell(txt=f'Right-most position at {datetime.timedelta(seconds=round(max_frame/fps,0))} (hours:minutes:seconds)',w=0,h=multicellHeight)

        pdf.image('max_frame.jpg',w=160,h=90)

        #The following code returns how much of the time they spent in the 'center' of the stage; that is, the middle 60% (sections 2,3 and 4)
        sectionOne = 0
        sectionTwo = 0
        sectionThree = 0
        sectionFour = 0
        sectionFive = 0
        for section in sections:
            if section == 1:
                sectionOne += 1
            if section == 2:
                sectionTwo += 1
            if section == 3:
                sectionThree += 1
            if section == 4:
                sectionFour += 1
            if section == 5:
                sectionFive += 1
        
        sectionOne = sectionOne/len(sections)*100
        sectionTwo = sectionTwo/len(sections)*100
        sectionThree = sectionThree/len(sections)*100
        sectionFour = sectionFour/len(sections)*100
        sectionFive = sectionFive/len(sections)*100

        pdf.set_font(family='Arial',style='B',size=p)

        pdf.multi_cell(txt=f'You were in the corners of the screen for {sectionOne+sectionFive}% of the presentation.',w=0,h=multicellHeight)

        pdf.multi_cell(txt='Graph of sections',w=0,h=multicellHeight)

        pdf.image('plot3.png',w=280,h=210)

        pdf.set_font(family='Arial',style='B',size=h1)

        pdf.multi_cell(txt='Hand gestures',w=0,h=40)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Graph of wrist positions ',w=0,h=multicellHeight)

        pdf.image('plot.png',w=280,h=210)

        pdf.set_font(family='Arial',style='B',size=h2)

        pdf.multi_cell(txt='Graph of wrist sections',w=0,h=multicellHeight)

        pdf.image('plot2.png',w=280,h=210)

        left_gestures = []

        pdf.multi_cell(txt='Left hand',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=p)

        #Adds all the hand gestures done by the left hand to a list

        for i in range(len(o_one_positive_intersections_left)):
            if i % 2 == 0:
                try:
                    left_gestures.append((o_one_positive_intersections_left[i],o_one_positive_intersections_left[i+1]))
                except:
                    pass

        for i in range(len(o_one_negative_intersections_left)):
            if i % 2 == 0:
                try:
                    left_gestures.append((o_one_negative_intersections_left[i],o_one_negative_intersections_left[i+1]))
                except:
                    pass

        left_gestures = sorted(left_gestures, key=lambda x: x[0])

        left_gestures_over_limit = []

        #Adds all the left hand gestures done for more than 5 seconds to another list

        for gesture in left_gestures:
            pdf.multi_cell(txt=f'Hand gesture from {gesture[0]} to {gesture[1]}',w=0,h=multicellHeight)
            if int(gesture[1]) - int(gesture[0]) >= 5:
                left_gestures_over_limit.append(gesture)

        for gesture in left_gestures_over_limit:
            pdf.multi_cell(txt=f'The gesture from {gesture[0]} to {gesture[1]} exceeded the 5-second recommended amount',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=h2)

        right_gestures = []

        pdf.multi_cell(txt='Right hand',w=0,h=multicellHeight)

        pdf.set_font(family='Arial',style='B',size=p)

        #Adds all the hand gestures done by the right hand to a list

        for i in range(len(o_one_positive_intersections_right)):
            if i % 2 == 0:
                try:
                    right_gestures.append((o_one_positive_intersections_right[i],o_one_positive_intersections_right[i+1]))
                except:
                    pass

        for i in range(len(o_one_negative_intersections_right)):
            if i % 2 == 0:
                try:
                    right_gestures.append((o_one_negative_intersections_right[i],o_one_negative_intersections_right[i+1]))
                except:
                    pass

        right_gestures = sorted(right_gestures, key=lambda x: x[0])

        right_gestures_over_limit = []

        #Adds all the right hand gestures done for more than 5 seconds to another list

        for gesture in right_gestures:
            pdf.multi_cell(txt=f'Hand gesture from {gesture[0]} to {gesture[1]}',w=0,h=multicellHeight)
            if int(gesture[1]) - int(gesture[0]) >= 5:
                right_gestures_over_limit.append(gesture)

        for gesture in right_gestures_over_limit:
            pdf.multi_cell(txt=f'The gesture from {gesture[0]} to {gesture[1]} exceeded the 5-second recommended amount',w=0,h=multicellHeight)

        pdf.multi_cell(txt=str(left_wrist_coords),w=0,h=multicellHeight)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')  # Use latin-1 here

        response = make_response(pdf_bytes)
        response.headers.set('Content-Type', 'application/pdf')
        response.headers.set('Content-Disposition', 'inline', filename=f'presentation-report-{datetime.datetime.now().strftime("%H%M%S")}.pdf')
        print("Process finished --- %s seconds ---" % (time.time() - start_time))
        #The PDF is loaded onto the user's tab
        return response

    #If the user goes back, they go back to the application!
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)

#And we're done