import os
import cv2
import argparse
import time
from model import ClassifyImage

def construct_row(roof_id, probs):
    row = []
    row.append(roof_id)
    for i in range(5):
        row.append("{:.8f}".format(probs[i]))
    return ','.join(row) + '\n'

def main():
    parser = argparse.ArgumentParser(description='Choose model for generating.')
    parser.add_argument(
        '-m', '--model', 
        type=str,
        help='Add models path...')
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]

# model_name = 'model.h5'
    labels_json_name = 'labels.json'
    output_csv = 'predict/output/predicted_test_data.csv'
    test_data_dir = 'stac/test_data'

    if not os.path.exists('predict/output'):
        os.makedirs('predict/output')

    model = ClassifyImage(  input_size=(224,224), 
                            model_path=('predict/model_folder/'+model_name),
                            label_path=('predict/model_folder/'+labels_json_name))

    with open(output_csv, 'w') as output_file:
        # Header
        output_file.write("id,concrete_cement,healthy_metal,incomplete,irregular_metal,other\n")

        all_images = len(os.listdir(test_data_dir))
        for i, img_file in enumerate(os.listdir(test_data_dir)):
            # exclude macOS dictionary file
            if not img_file.startswith('.'):
                fps_start = time.time()
                
                img = cv2.imread(test_data_dir+ '/' + img_file, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pred_probabilities = model.predict(img, all_probabilities=True)
                # pred_probabilities example -> [0.27870926, 0.203701, 0.05302142, 0.40451434, 0.06005398]
                #print(pred_probabilities)

                predicted_row_str = construct_row(img_file.split('.')[0], pred_probabilities)

                output_file.write(predicted_row_str)
            
                fps_end = time.time()
                print('processed -> {}/{} with {:.2f} fps'.format(i+1,all_images, 1/(fps_end-fps_start)))

if __name__ == "__main__":
    main()