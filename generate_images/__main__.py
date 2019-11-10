import os
import sys
import gc
import numpy as np
from generate_images import generate_images
import time


def validate_urls(urls):
    urls = np.array(urls).flatten()
    for url in urls:
        if not os.path.exists(url):
            print("ERROR: Following file path does not exists\n  '{}'\nReturning.".format(url),
                  file=sys.stderr)
            return False
    return True


def main():
    process_start_time = time.time()

    URLS_BASES = [
        "stac/colombia/borde_rural",
        "stac/colombia/borde_soacha",
        "stac/guatemala/mixco_1_and_ebenezer",
        "stac/guatemala/mixco_3",
        "stac/st_lucia/dennery"
    ]
    target_dir = "stac/training_data"

    ortho_extension = "{}_ortho-cog.tif"
    # ortho_extension = "{}_ortho-cog-thumbnail.png"
    map_extension = "{}-imagery.json"
    labels_extension = "train-{}.geojson"
    img_format = "tiff"

    URLS = [(
        base + "/" + ortho_extension.format(base.split("/")[-1]),
        base + "/" + map_extension.format(base.split("/")[-1]),
        base + "/" + labels_extension.format(base.split("/")[-1])
    ) for base in URLS_BASES]

    # if a file does not exists, exit program
    if not validate_urls(URLS):
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for url in URLS:
        generate_images(url, to_folder=target_dir, img_format=img_format)
        gc.collect()

    print("Finished in: {:.3f}s".format(time.time()-process_start_time))


if __name__ == "__main__":
    main()
