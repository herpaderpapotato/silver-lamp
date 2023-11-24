import os
import time
import numpy as np
from glob import glob
from collections import deque
from keras_cv_attention_models import efficientnet
import json
import random
import cv2
import requests
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)
from tensorflow import keras
from tensorflow.keras import layers

image_frames = 60
image_size = 384
video_name = ''
model_name = ''
model = None

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video', help='video file to process')
parser.add_argument('--model', help='model file to use')
args = parser.parse_args()
if args.video:
    video_name = args.video
if args.model:
    model_name = args.model



def download_model():
    model_url = "https://huggingface.co/herpaderpapotato/sixty_small_body_ryhthm_time_bidirectional/resolve/main/effv2s_60in_60out_129e.h5?download=true"
    model_name = model_url.split("/")[-1].split("?")[0]
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        print(f"Downloading model {model_name} from {model_url}...")
        r = requests.get(model_url, allow_redirects=True)
        open(model_path, "wb").write(r.content)
    else:
        print(f"Model {model_name} already downloaded.")


models = glob('models/*.h5')
if not len(models) > 0 and model_name == '':
    print('no model specified, downloading default model')
    download_model()

if not os.path.exists('models'):
    os.makedirs('models')

if video_name == '':
    videos = glob('*.mp4')
    if len(videos) > 0:
        videos.sort(key=os.path.getmtime)
        video_name = videos[-1]
        print('loading video: ' + video_name)
    else:
        print('no video specified')
        exit()
else:
    if not os.path.exists(video_name):
        print('video does not exist', video_name)
        exit()


def load_model():
    global model, model_name
    if model_name == '':
        print('no model specified')
        models = glob('models/*.h5')
        if len(models) > 0:
            models.sort(key=os.path.getmtime)
            model_name = models[-1]
            print('loading model: ' + model_name)
            model = keras.models.load_model(model_name)
        else:
            download_model()
            models = glob('models/*.h5')
            if len(models) > 0:
                models.sort(key=os.path.getmtime)
                model_name = models[-1]
                print('loading model: ' + model_name)
                model = keras.models.load_model(model_name)
            else:
                print('model does not exist', 'models/' + model_name)
                exit()

    else:
        if not os.path.exists('models/' + model_name):
            print('model does not exist', 'models/' + model_name)
            exit()
        print('loading model: models/' + model_name)
        model = keras.models.load_model('models/' + model_name)

cap = cv2.VideoCapture(video_name)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('fps', fps)
print('frame_count', frame_count)
print('frame_width', frame_width)
print('frame_height', frame_height)

frame_delay_multiplier = (1 / fps) * 1000 / 2
frame_delay_base = 2
frame_delay = max(frame_delay_base * frame_delay_multiplier,1)

predict = False

prediction_frames = deque(maxlen=image_frames)
prediction_frames_crop = deque(maxlen=image_frames)
prediction_frames_poi = deque(maxlen=image_frames)
prediction_frames_times = deque(maxlen=image_frames)
prediction_frames_frame_numbers = deque(maxlen=image_frames)
prediction_frames_times_poi = deque(maxlen=image_frames)
prediction_frames_frame_numbers_poi = deque(maxlen=image_frames)

poi_pct = 0.2
poi_x = 0
poi_y = 0
poi_offset = max(int(frame_width * poi_pct) // 2, image_size // 2)
poi = False
display = True
hold = False

def on_mouse(event, x, y, flags, param):
    global poi_x, poi_y, poi
    if event == cv2.EVENT_LBUTTONDOWN:
        poi_x = int(x / image_size * frame_width)
        poi_y = int(y / image_size * frame_height)
        if poi_x < poi_offset:
            poi_x = poi_offset
        if poi_x > frame_width - poi_offset:
            poi_x = frame_width - poi_offset
        if poi_y < poi_offset:
            poi_y = poi_offset
        if poi_y > frame_height - poi_offset:
            poi_y = frame_height - poi_offset
        poi = True

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', on_mouse)
print(frame_delay)

prediction_times = []
prediction_logged = []
prediction_frame_numbers = []
prediction_times_poi = []
prediction_frame_numbers_poi = []

try:
    ret, frame = cap.read()
    display_frame = frame.copy()[0:frame_height, 0:frame_width]
    cv2.imshow('frame', display_frame)
    key = cv2.waitKey(int(frame_delay))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if display:
            display_frame = cv2.resize(frame.copy()[0:frame_height, 0:frame_width], (image_size, image_size))
            if poi:
                display_poi_x = poi_x / frame_width * image_size
                display_poi_y = poi_y / frame_height * image_size
                display_poi_offset = poi_offset / frame_width * image_size
                cv2.circle(display_frame, (int(display_poi_x), int(display_poi_y)), int(display_poi_offset), (0, 0, 255), 2)
                cv2.rectangle(display_frame, (int(display_poi_x - display_poi_offset), int(display_poi_y - display_poi_offset)), (int(display_poi_x + display_poi_offset), int(display_poi_y + display_poi_offset)), (0, 0, 255), 2)
            if predict:
                # show a red circle in the top left corner
                cv2.circle(display_frame, (0, 0), 10, (0, 0, 255), 10)
            cv2.imshow('frame', display_frame)
        if hold:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(int(frame_delay))
        if key == ord('q'):
            break
        if key == ord('p'):
            predict = True
        if key == ord('d'):
            display = not display
        if key == ord('h'):
            hold = not hold
        if key == ord(','):
            frame_delay_multiplier -= 1
            frame_delay = max(frame_delay_base * frame_delay_multiplier,1)
        if key == ord('.'):
            frame_delay_multiplier += 1
            frame_delay = max(frame_delay_base * frame_delay_multiplier,1)
        if key == ord('['):
            poi_pct -= 0.01
            poi_pct = max(poi_pct, 0.05)
            poi_offset = max(int(frame_width * poi_pct) // 2, image_size // 2)
        if key == ord(']'):
            poi_pct += 0.01
            poi_pct = min(poi_pct, 1)
            poi_offset = max(int(frame_width * poi_pct) // 2, image_size // 2)
            # need to add some code to make sure poi_x and poi_y are still within bounds based on the offset or growing the pct could cause an error


        if poi:
            poi_frame = frame.copy()[poi_y - poi_offset:poi_y + poi_offset, poi_x - poi_offset:poi_x + poi_offset]
            if poi_frame.shape[0] != image_size or poi_frame.shape[1] != image_size:
                poi_frame = cv2.resize(poi_frame, (image_size, image_size))
            poi_frame = cv2.cvtColor(poi_frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow('poi', poi_frame)
            # key = cv2.waitKey(1)
            prediction_frames_poi.append(poi_frame)
            prediction_frames_times_poi.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            prediction_frames_frame_numbers_poi.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

        crop_frame = frame.copy()[frame_height // 2:frame_height, frame_width // 4 : frame_width - frame_width // 4]
        if crop_frame.shape[0] != image_size or crop_frame.shape[1] != image_size:
            crop_frame = cv2.resize(crop_frame, (image_size, image_size))
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('crop', crop_frame)
        # key = cv2.waitKey(1)
        prediction_frames_crop.append(crop_frame)
        prediction_frames_times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        prediction_frames_frame_numbers.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = frame[0:frame_height, 0:frame_width]
        frame = cv2.resize(frame, (image_size, image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('mainframe', frame)
        # key = cv2.waitKey(1)
        prediction_frames.append(frame)
        if predict:
            if model is None:
                load_model()

        if len(prediction_frames) == image_frames:
            if predict:
                prediction_times.extend(prediction_frames_times)
                prediction_frame_numbers.extend(prediction_frames_frame_numbers)
                if poi and len(prediction_frames_poi) == len(prediction_frames):
                    prediction_times_poi.extend(prediction_frames_times_poi)
                    prediction_frame_numbers_poi.extend(prediction_frames_frame_numbers_poi)
                if poi and len(prediction_frames_poi) == image_frames:
                    prediction_frames_array = np.array([np.array(prediction_frames), np.array(prediction_frames_crop), np.array(prediction_frames_poi)])
                else:
                    prediction_frames_array = np.array([np.array(prediction_frames), np.array(prediction_frames_crop)])
                predictions = model.predict(prediction_frames_array)
                prediction_logged.append(predictions)
                print(predictions)
                prediction_frames.clear()
                prediction_frames_crop.clear()
                prediction_frames_poi.clear()

        if key == ord('v'):
            cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 60000)
        if key == ord('b'):
            cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 360000)
        if key == ord('c'):
            cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 60000)
        if key == ord('x'):
            cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 360000)
        if key == ord('z'):
            cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 10000)
except KeyboardInterrupt:
    pass
cv2.destroyAllWindows()

if len(prediction_logged) > 0:
    # dump to npy file
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    prediction_logged_original = []
    prediction_logged_cropped = []
    prediction_logged_poi = []
    for prediction in prediction_logged:
        #print(len(prediction[0]))
        for i in range(len(prediction[0])):
            prediction_logged_original.append(prediction[0][i])
            prediction_logged_cropped.append(prediction[1][i])
            try:
                prediction_logged_poi.append(prediction[2][i])
            except IndexError:
                pass

    funscript_filename = os.path.basename(video_name) + '.' + time.strftime("%Y%m%d-%H%M%S") + '.funscript'
    funscript_json_data = {}
    funscript_json_data['version'] = '1.0'
    funscript_json_data['inverted'] = False
    funscript_json_data['range'] = 100
    funscript_json_data['actions'] = []
    for i in range(len(prediction_logged_original)):
        funscript_json_data['actions'].append({'at': int(prediction_times[i]), 'pos': int(prediction_logged_original[i] * 100)})
    with open('predictions/' + funscript_filename, 'w') as outfile:
        json.dump(funscript_json_data, outfile)
    
    funscript_filename = os.path.basename(video_name) + '.crop.' + time.strftime("%Y%m%d-%H%M%S") + '.funscript'
    funscript_json_data = {}
    funscript_json_data['version'] = '1.0'
    funscript_json_data['inverted'] = False
    funscript_json_data['range'] = 100
    funscript_json_data['actions'] = []
    for i in range(len(prediction_logged_cropped)):
        funscript_json_data['actions'].append({'at': int(prediction_times[i]), 'pos': int(prediction_logged_cropped[i] * 100)})
    with open('predictions/' + funscript_filename, 'w') as outfile:
        json.dump(funscript_json_data, outfile)

    funscript_filename = os.path.basename(video_name) + '.poi.' + time.strftime("%Y%m%d-%H%M%S") + '.funscript'
    funscript_json_data = {}
    funscript_json_data['version'] = '1.0'
    funscript_json_data['inverted'] = False
    funscript_json_data['range'] = 100
    funscript_json_data['actions'] = []
    for i in range(len(prediction_logged_poi)):
        try:
            funscript_json_data['actions'].append({'at': int(prediction_times[i]), 'pos': int(prediction_logged_poi[i] * 100)})
        except IndexError:
            pass
    with open('predictions/' + funscript_filename, 'w') as outfile:
        json.dump(funscript_json_data, outfile)


    
    