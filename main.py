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
import os

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

data=[]
# for i in range(1,10):
#     temp=[]
#     example_path = 'dataset/seq_00000'+str(i)+".jpg"
#     count_persons1 =  count_persons(example_path,detector)
#     temp.append(i)
#     temp.append(example_path)
#     temp.append(count_persons1)
#     data.append(temp)

folder="dataset"
i=1
for filename in os.listdir(folder):
    print(i)
    if(i==40):
        break
    temp=[]
    example_path = 'dataset/'+filename
    count_persons1 =  count_persons(example_path,detector)
    temp.append(i)
    temp.append(example_path)
    if(count_persons1<10):
        temp.append(count_persons1)
        temp.append("non crowd")
    else:
        temp.append(count_persons1)
        temp.append("crowd")

    
    data.append(temp)
    i=i+1
print("")
df = pd.DataFrame(data, columns = ['id', 'path','count','crowd/non crowd'])
print(df)
stats = df.describe()
plt.hist(df['count'], bins=20)
plt.axvline(stats.loc['mean', 'count'], label='Mean value', color='green')
plt.legend()
plt.xlabel('Number of people')
plt.ylabel('Frequency')
plt.title('Target Values')
plt.show()


sample = df.sample(frac=0.1)
start = time.perf_counter()
objects = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = [executor.submit(count_persons, path, detector, 0.23) for path in sample['path']]
    for f in tqdm(concurrent.futures.as_completed(results)):
        objects.append(f.result())

finish = time.perf_counter()
print(f'Finished in {round(finish - start, 2)} second(s).')
sample['prediction'] = objects
print(sample)


sample['mae'] = (sample['count'] - sample['prediction']).abs()
sample['mse'] = sample['mae'] ** 2

print(f'MAE = {sample["mae"].mean()}\nMSE = {sample["mse"].mean()}')
plt.hist(sample['mae'], bins=20)
plt.title('Absolute Errors')
plt.show()
