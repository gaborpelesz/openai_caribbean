import numpy as np
import cv2
import json
import os
from bigimage import Bigimage, Reader
from preprocessing.perspective_transform import get_rotated_bounding_rect, four_point_transform
from preprocessing.image_resize import image_resize


def load_json_file(url):
    with open(url) as f:
        datastring = f.read()

    return json.loads(datastring)


def generate_images(image_url_tuple, to_folder, img_format="tif"):
    """
    Params:
        image_url_tuple: URL tuple, where tuples are path to the
            following file -> (image.<tif || png>, image.json, roof_roofs.json)
    """
    image_url, image_json_url, roof_roof_url = image_url_tuple

    # load image bounding box
    image_json = load_json_file(image_json_url)
    image_bbox = [image_json["bbox"][:2], image_json["bbox"][2:]]

    # load roof roof data
    roofs_data = load_json_file(roof_roof_url)

    # read image
    image = None
    if img_format == "tiff" or img_format == "tif":
        image = Reader.read_tif(image_url)
    elif img_format == "png" or img_format == "jpg":
        image = Reader.read_img(image_url)

    # creating bigimage object
    bigimage = Bigimage(image, image_bbox)

    # saving length so we can print where we are
    len_features_list = len(roofs_data["features"])

    for i, feature in enumerate(roofs_data["features"]):
        print(
            "{} -> ({} / {})".format(image_url.split("/")[-2], i+1, len_features_list))
        # we don't deal with multipolygons yet
        if feature["geometry"]["type"] == "Polygon":
            # access raw coordinates from json data
            raw_roof_coordinates = feature["geometry"]["coordinates"][0]
            # access roof ID and roof material
            roof_id, roof_material = (
                feature["properties"]["id"], feature["properties"]["roof_material"])

            roof_coordinate_pixels = bigimage.coords_to_pixels(
                raw_roof_coordinates)

            roof_pixels = get_rotated_bounding_rect(
                roof_coordinate_pixels, INFO=False)

            roof_top_left = tuple(np.min(roof_pixels, axis=0).astype(
                np.int32).tolist())
            roof_bottom_right = tuple(np.max(roof_pixels, axis=0).astype(
                np.int32).tolist())

            # defines the additional size of the rectangle cut from the image
            # the roof is inside that rectangle
            rect_size = 20

            # cut a rectangular shape from the image where the roof is in the middle
            decreased_image = image_cut_rectangle(image, ([roof_top_left[0]-rect_size, roof_top_left[1]-rect_size],
                                                          [roof_bottom_right[0]+rect_size, roof_bottom_right[1]+rect_size]))

            # refactor roof pixels to match the decreased
            roof_pixels[:, 0] = roof_pixels[:, 0] - \
                (roof_top_left[0]-rect_size)
            roof_pixels[:, 1] = roof_pixels[:, 1] - \
                (roof_top_left[1]-rect_size)

            # 2d perspective transform
            transformed_roof_image = four_point_transform(
                decreased_image, roof_pixels)

            # scale image to 224x224
            # transformed_roof_image = cv2.resize(
            #    transformed_roof_image, (224, 224), interpolation=cv2.INTER_CUBIC)

            # write image to destination
            cv2.imwrite('{}/{}-{}.png'.format(to_folder, roof_id,
                                              roof_material), transformed_roof_image)


def image_cut_rectangle(image, rectangle_vertices):
    roof_top_left = list(rectangle_vertices[0])
    roof_bottom_right = list(rectangle_vertices[1])
    if roof_top_left[0] < 0:
        roof_top_left[0] = 0
    if roof_top_left[1] < 0:
        roof_top_left[1] = 0
    if roof_bottom_right[0] > image.shape[1]:
        roof_bottom_right[0] = image.shape[1]
    if roof_bottom_right[1] > image.shape[0]:
        roof_bottom_right[1] = image.shape[0]

    return image[roof_top_left[1]:roof_bottom_right[1], roof_top_left[0]:roof_bottom_right[0], :].copy()
