import os
import numpy as np
import json

def validate_urls(urls):
    urls = np.array(urls).flatten()
    for url in urls:
        if not os.path.exists(url):
            print("ERROR: Following file path does not exists\n  '{}'\nReturning.".format(url),
                  file=sys.stderr)
            return False
    return True

def load_json_file(url):
    with open(url) as f:
        datastring = f.read()

    return json.loads(datastring)

def order_reference():
    URLS_BASES = [
        "stac/colombia/borde_rural",
        "stac/colombia/borde_soacha",
        "stac/guatemala/mixco_1_and_ebenezer",
        "stac/guatemala/mixco_3",
        "stac/st_lucia/dennery"
    ]

    labels_extension = "test-{}.geojson"

    URLS = [[
        base + "/" + labels_extension.format(base.split("/")[-1])
    ] for base in URLS_BASES]

    if not validate_urls(URLS):
        return

    ordered_ids = []

    for url in URLS:
        roofs_data = load_json_file(url[0])
        for feature in roofs_data["features"]:
            ordered_ids.append(feature["id"])

    return ordered_ids

def sort_rows():
    reference = order_reference()

    with open("predict/output/unsorted.csv", "r") as csv_file:
        # split to rows and exclude header
        data = csv_file.read().split('\n')
        header = data[0]
        data_rows = data[1:]

    data_id_dict = {row.split(',')[0]: row.split(',')[1:] for row in data_rows}
    
    output = []
    output.append(header)

    for feature_id in reference:
        output.append(','.join([feature_id] + data_id_dict[feature_id]))

    with open("predict/output/sorted_test_data_prediction.csv", "w") as csv_file:
        csv_file.write('\n'.join(output))

sort_rows()