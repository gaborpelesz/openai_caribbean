import json
import numpy as np
import sys
from tensorflow.compat.v2.image import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

class ClassifyImage:
    def __init__(self,  input_size=(224,224),
                        model_path='app/model/object_classification.h5',
                        label_path='app/model/labels.json'):
        self.model = load_model(model_path)

        # load labels
        with open(label_path) as label_file:
            data = label_file.read().replace('\n', '').replace("'", '"')
            json_data = json.loads(data)

        self.labels = json_data
        self.input_size = input_size

    def predict(self, img):
        # resize the image
        img = np.array(resize(np.array([img]), self.input_size)[0])
        # preprocess the input image for ResNet50
        preprocessed_img = preprocess_input(img)

        predictions = self.model.predict(np.array([preprocessed_img]))[0]
        # selecting the prediction with the highest probability
        highest_pred_index = np.where(predictions == max(predictions))[0][0]

        # map the correct label with the highest prediction's index
        for label, label_num in self.labels.items():
            if label_num == highest_pred_index:
                return (label, max(predictions))
        raise Exception("Couldn't find matching prediction index and label number...")