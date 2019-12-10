import cv2
import os
import numpy as np
from keras.models import load_model

from keras.backend import sigmoid

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

swish = Activation(swish)
swish.__name__ = 'swish'

get_custom_objects().update({'swish': swish})

def construct_row(roof_id, probs):
    row = []
    row.append(roof_id)
    for i in range(5):
        row.append("{:.8f}".format(probs[0][i]))
    return ','.join(row) + '\n'

labels_json_name = 'predict/model_folder/labels.json'
output_csv = 'predict/output/unsorted.csv'
test_data_dir = 'stac/datasets/test_data'
model_path = 'predict/model_folder/(best)augmented_v2_VAL_LOSS_0.0003_VAL_ACC_0.7496.h5'

input_size = (280,280)

model = load_model(model_path)

with open(output_csv, 'w') as output_file:
    output_file.write("id,concrete_cement,healthy_metal,incomplete,irregular_metal,other\n")
    all_img_num = len(os.listdir(test_data_dir))

    for i, img_file in enumerate(os.listdir(test_data_dir)):
        if not img_file.startswith('.'):
            img = cv2.imread(test_data_dir+ '/' + img_file, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, input_size, interpolation = cv2.INTER_AREA)

            pred_probabilities = model.predict(np.array([img]))
            predicted_row_str = construct_row(img_file.split('.')[0], pred_probabilities)
            output_file.write(predicted_row_str)

            print('finished', i+1, '/', all_img_num)