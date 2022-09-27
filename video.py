import cv2
import time
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import PIL
from PIL.ImageDraw import Draw


MODEL_PATH = 'https://tfhub.dev/tensorflow/efficientdet/d0/1'

detector = hub.load(MODEL_PATH)




def detect_objects(path: str, model) -> dict:
    image_tensor = tf.image.decode_jpeg(
        tf.io.read_file(path), channels=3)[tf.newaxis, ...]
    return model(image_tensor)


def count_persons(path: str, model, threshold=0.23) -> int:
   
    results = detect_objects(path, model)
    # Class ID 1 = "person"
    return (results['detection_classes'].numpy()[0] == 1)[np.where(
        results['detection_scores'].numpy()[0] > threshold)].sum()


def draw_bboxes(image_path, data: dict, threshold=0.23) -> PIL.Image:
   
    image = PIL.Image.open(image_path)
    print(image)
    draw = Draw(image)

    im_width, im_height = image.size

    boxes = data['detection_boxes'].numpy()[0]
    classes = data['detection_classes'].numpy()[0]
    scores = data['detection_scores'].numpy()[0]

    for i in range(int(data['num_detections'][0])):
        if classes[i] == 1 and scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                      width=4, fill='red')

    return image

cap= cv2.VideoCapture('Video/1.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break


    cv2.imwrite('video frame/'+str(i)+'.jpg',frame)

    results = detect_objects('video frame/'+str(i)+'.jpg', detector)
    
    example_path='video frame/'+str(i)+'.jpg'
    count_persons1 =  count_persons(example_path,detector)
    image =  draw_bboxes(example_path, results)
    # cv2.imwrite('video frame/'+str(i)+'.jpg',image)
    image.save(example_path)

    image = cv2.imread(example_path)
    window_name = 'image'
    
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow(window_name, image)
    i+=1
 
cap.release()
cv2.destroyAllWindows()