import os
import sys
import gc
import numpy as np
import time
import argparse
from extract_images import extract_images

def get_URLS(is_test=False):
    URLS_BASES = [
        "stac/colombia/borde_rural",
        "stac/colombia/borde_soacha",
        "stac/guatemala/mixco_1_and_ebenezer",
        "stac/guatemala/mixco_3",
        "stac/st_lucia/dennery"
    ]

    ortho_extension = "{}_ortho-cog.tif"
    labels_extension = ""
    if not is_test:
        labels_extension = "train-{}.geojson"
    else:
        labels_extension = "test-{}.geojson"

    URLS = [(
        base + "/" + ortho_extension.format(base.split("/")[-1]),
        base + "/" + labels_extension.format(base.split("/")[-1])
    ) for base in URLS_BASES]
    return URLS


def validate_urls(urls):
    urls = np.array(urls).flatten()
    for url in urls:
        if not os.path.exists(url):
            print("ERROR: Following file path does not exists\n  '{}'\nReturning.".format(url),
                  file=sys.stderr)
            return False
    return True


def create_label_folders(target):
    folders_to_create = [
        os.path.join(target, 'concrete_cement'),
        os.path.join(target, 'healthy_metal'),
        os.path.join(target, 'incomplete'),
        os.path.join(target, 'irregular_metal'),
        os.path.join(target, 'other')
    ]
    for f in folders_to_create:
        if not os.path.exists(f):
            os.makedirs(f)


def main():
    ### ARGUMENT PARSER ###
    parser = argparse.ArgumentParser(description='Choose wether to extract the training or the test images')
    parser.add_argument(
        '-t', '--test', 
        action='store_true',
        help='Switch to test images')
    args = parser.parse_args()

    ### TARGET DIRECTORY ###
    EXTRACT_TEST = args.test
    EXTRACT_TRAINING = not EXTRACT_TEST

    target_dir = ""
    if EXTRACT_TRAINING:
        target_dir = "stac/datasets/training_data"
    if EXTRACT_TEST:
        target_dir = "stac/datasets/test_data"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    ### CREATE LABEL DIRECTORIES ###
    create_label_folders(target_dir)

    ### CALCULATE RUNTIME ###
    process_start_time = time.time()

    URLS = get_URLS(EXTRACT_TEST)
    
    # if a file does not exists, exit program
    if not validate_urls(URLS):
        return

    for tif_path,geojson_path in URLS:
        extract_images( (tif_path,geojson_path), to_folder=target_dir, test_data=EXTRACT_TEST)
        gc.collect()

    print("Finished in: {:.3f}s".format(time.time()-process_start_time))


if __name__ == "__main__":
    main()
