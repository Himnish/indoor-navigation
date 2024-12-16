# -*- coding: utf-8 -*-
"""
Object Detection and recognition, with a voice warning system for close distances
"""

import numpy as np
import os
import sys
import time
import tarfile
import tempfile
import uuid
import threading
from queue import Queue

import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/Users/himnishjain/anaconda3/bin/pytesseract'

import urllib.request as allib
import ssl
import zipfile
import pyttsx3
import cv2

import tensorflow as tf
sys.path.append('/Users/himnishjain/Desktop/CS445_Final/Blind-Assistance-Object-Detection-and-Navigation/models/research')
sys.path.append('/Users/himnishjain/Desktop/CS445_Final/Blind-Assistance-Object-Detection-and-Navigation/models/research/slim')

from matplotlib import pyplot as plt
from PIL import Image
from gtts import gTTS
from playsound import playsound

from utils import label_map_util
from utils import visualization_utils as vis_util

# Use macOS SSL fix if needed
ssl._create_default_https_context = ssl._create_unverified_context

def speak_async(text):
    """Generate TTS audio and play in a separate daemon thread (non-blocking)."""
    def worker():
        tts = gTTS(text=text, lang='en')
        tmp_filename = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")
        tts.save(tmp_filename)
        playsound(tmp_filename)
        os.remove(tmp_filename)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

# TensorFlow model setup
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = "https://download.tensorflow.org/models/object_detection/"
    print('Downloading the model...')
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print('Download complete.')
else:
    print('Model already exists.')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True
)

# Initialize TTS engine + background queue (if needed)
engine = pyttsx3.init()
tts_queue = Queue()
def tts_worker():
    """Continuously run TTS in background."""
    while True:
        message = tts_queue.get()
        if message is None:
            break
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print("TTS engine error:", e)
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

cap = cv2.VideoCapture(0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

last_warning_time = 0
WARNING_COOLDOWN = 5.0  # seconds

threshold = 0.5  # Confidence threshold
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
        while True:
            ret, image_np = cap.read()
            if not ret:
                print("Frame capture failed. Retrying...")
                continue

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
            classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection
            (boxes_out, scores_out, classes_out, num_detections_out) = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                feed_dict={image_tensor: image_np_expanded}
            )

            # Convert outputs to a nicer shape
            boxes_out = np.squeeze(boxes_out)
            scores_out = np.squeeze(scores_out)
            classes_out = np.squeeze(classes_out).astype(np.int32)
            num_detections_out = int(np.squeeze(num_detections_out))

            detections = []
            height, width, _ = image_np.shape

            for i in range(num_detections_out):
                score = scores_out[i]
                if score > threshold:
                    cls = classes_out[i]
                    name = category_index.get(cls, {}).get('name', 'Unknown')
                    box = boxes_out[i]  # [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = box
                    # Arbitrary "distance" heuristic
                    distance = round(((1 - (xmax - xmin)) ** 4), 2)

                    detections.append((name, score, distance, (ymin, xmin, ymax, xmax)))

            # Sort by distance ascending (closest first)
            detections.sort(key=lambda x: x[2])

            # Warn if first detection is too close
            if detections and detections[0][2] <= 0.1:
                current_time = time.time()
                if current_time - last_warning_time > WARNING_COOLDOWN:
                    closest_name, closest_score, closest_distance, _ = detections[0]
                    warning_message = (
                        f"Warning, a {closest_name} is just {closest_distance} meters away!"
                    )
                    print(warning_message)
                    speak_async(warning_message)
                    last_warning_time = current_time

            # 1) Draw only bounding boxes (no text) via TF's utility:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes_out,
                classes_out,
                scores_out,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=threshold,
                skip_scores=True,  # <--- we skip the default tiny score text to replace with our own version which is readable
                skip_labels=True   # <--- skip the default label text for the same reason
            )

            # 2) Manually overlay bigger text, including distance:
            for name, score, distance, (ymin, xmin, ymax, xmax) in detections:
                # Convert from normalized coords to pixel coords
                (left, right, top, bottom) = (xmin * width, xmax * width,
                                              ymin * height, ymax * height)
                
                label_text = f"{name} {score*100:.1f}% dist={distance}"
                # Position the text above the box (or inside if you'd like)
                text_pos = (int(left), max(int(top) - 10, 20))
                # Adjust font scale and thickness as desired
                cv2.putText(
                    image_np,
                    label_text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,              
                    (0, 255, 0),      
                    3,                
                    cv2.LINE_AA
                )

            cv2.imshow('Live Detection', cv2.resize(image_np, (1024, 768)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

# Cleanup
cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)
tts_queue.join()
