import rasterio
import numpy as np
import cv2
import time


class Bigimage:
    def __init__(self, bigimage, bbox):
        self.bigimage = bigimage
        self.shape = bigimage.shape
        self.bbox = bbox
        # ( bottom right x coord - top left x coord ) / image x pixels
        self.longitude_unit = abs((bbox[1][0]-bbox[0][0])) / bigimage.shape[1]

        # ( bottom right y coord - top left y coord ) / image y pixels
        self.latitude_unit = abs((bbox[1][1]-bbox[0][1])) / bigimage.shape[0]

    def latitude_to_pix(self, latitude):  # ~ 4.5xxxxxx
        pix = round(abs(latitude-self.bbox[1][1]) / self.latitude_unit)
        if self.bbox[1][1] < 0:
            pix = self.shape[0] - pix
        return pix

    def longitude_to_pix(self, longitude):  # ~ -74.1xxxxxx
        pix = round(abs(longitude-self.bbox[1][0]) / self.longitude_unit)
        #print("{}, {} -> {}".format(self.shape[1], longitude, pix))
        if self.bbox[1][0] < 0:
            pix = self.shape[1] - pix
        return pix

    def house_bbox(self, polygon_vertices):
        np.array(polygon_vertices)
        house_bbox = [
            np.min(polygon_vertices, axis=0).tolist(),
            np.max(polygon_vertices, axis=0).tolist()
        ]

        lat_pixels = (self.latitude_to_pix(
            house_bbox[0][1]), self.latitude_to_pix(house_bbox[1][1]))
        lon_pixels = (self.longitude_to_pix(
            house_bbox[0][0]), self.longitude_to_pix(house_bbox[1][0]))
        house_bbox_pixels = [
            (min(lon_pixels), min(lat_pixels)),
            (max(lon_pixels), max(lat_pixels))
        ]

        return house_bbox_pixels

    def coords_to_pixels(self, coords, long_lat=True):
        coords = np.array(coords, dtype="float64")
        long_lat = 0 if long_lat else 1
        pixels = []

        if coords.shape[-1] == 2:
            for i, n in enumerate(np.nditer(coords)):

                if (i+long_lat) % 2 == 0:
                    pixels.append(self.longitude_to_pix(n))
                else:
                    pixels.append(self.latitude_to_pix(n))

        return np.array(pixels, np.float32).reshape(len(pixels)//2, 2)


class Reader:
    @staticmethod
    def read_img(src_path):
        assert src_path.endswith('.png') or src_path.endswith('.jpg'), \
            "Error reading file with extension '{}' as a .jpg or .png file.".format(
                src_path.split('.')[-1])

        return cv2.imread(src_path, 1)

    @staticmethod
    def read_tif(src_path):
        assert src_path.endswith('.tif'), \
            "Error reading file with extension '{}' as a .tif file.".format(
                src_path.split('.')[-1])

        print("Start reading .tif image.")
        start_reading = time.time()
        with rasterio.open(src_path) as src:
            channels = src.read()
        end_reading = time.time()
        print("Reading was successful. ({:.3f} sec)\n".format(
            end_reading-start_reading))

        return Reader.channels_to_ndarray(channels)

    @staticmethod
    def channels_to_ndarray(channels):
        print("Converting channels to numpy array.")
        start = time.time()

        r, g, b = channels[:3]
        r = r.reshape(r.shape[0], r.shape[1], 1)
        g = g.reshape(g.shape[0], g.shape[1], 1)
        b = b.reshape(b.shape[0], b.shape[1], 1)

        print("Finished converting. ({:.5f} sec)\n".format(time.time()-start))
        return np.dstack((b, g, r))
