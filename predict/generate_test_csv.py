import os
import cv2
from model import ClassifyImage

def construct_row(roof_type, prob):
    labels = [  "concrete_cement",
                "healthy_metal",
                "incomplete",
                "irregular_metal",
                "other"]
    roof_idx = labels.index(roof_type)

    row = []
    row.append(roof_type)
    for i in range(5):
        if roof_idx == i:
            row.append(str(prob))
        else:
            row.append('0.0')
    return ','.join(row) + '\n'

model_name = 'model.h5'
labels_json_name = 'labels.json'
output_csv = 'predict/output/predicted_test_data.csv'
test_data_dir = 'stac/test_data'

if not os.path.exists('predict/output'):
    os.makedirs('predict/output')

model = ClassifyImage(  input_size=(224,224), 
                        model_path=('predict/model_folder/'+model_name),
                        label_path=('predict/model_folder/'+labels_json_name))

with open(output_csv, 'w') as output_file:
    all_images = len(os.listdir(test_data_dir))
    for i, img_file in enumerate(os.listdir(test_data_dir)):
        print('processing -> {}/{}'.format(i+1,all_images))
        # exclude macOS dictionary file
        if not img_file.startswith('.'):
            img = cv2.imread(test_data_dir+ '/' + img_file, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_roof_type, pred_probability = model.predict(img)

            predicted_row_str = construct_row(pred_roof_type, pred_probability)

            output_file.write(predicted_row_str)