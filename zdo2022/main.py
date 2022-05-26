from __future__ import annotations
from configparser import RawConfigParser
from itertools import accumulate
from podpurne_funkce import get_image_files
from podpurne_funkce import scale_image
from skimage.transform import (probabilistic_hough_line, hough_circle)
from skimage.feature import (canny, peak_local_max)
from skimage.draw import (ellipse_perimeter, circle_perimeter)

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import skimage.morphology


class InstrumentTracker():
    def __init__(self):
        pass

    def create_annotation_output(self, filename, frame_id, object_id, x_px, y_px, annotation_timestamp):
        """
        :param filename: names of processed files
        :param frame_id: processed frames
        :param object_id: processed objects in order of listed points
        :param x_px: x coordinates of discovered points
        :param y_px: y coordinated of discovered points
        :param annotation_timestamp: time of annotation
        :return: annnotations
        """
        annotation={
            "filename": filename,
            "frame_id": frame_id,
            "object_id": object_id,
            "x_px": x_px,
            "y_px": y_px,
            "annotation_timestamp": annotation_timestamp,
        }

        print(annotation)

        return annotation

    def predict(self, video_dirname):
        """
        :param video_dirname: name of the directory containing the .png files the video consists of
        :return: annnotations
        """
        filename = []
        frame_id = []
        object_id = []
        x_px = []
        y_px = []

        filename.append(video_dirname)

        image_files = get_image_files(video_dirname + "/images")
        #print(image_files)

        for i in range(0, len(image_files), 2):
            image_filename = video_dirname + "/images/" + image_files[i]
            frame = cv.imread(image_filename)
            frame = scale_image(frame, 0.25)

            #image_filename = video_dirname + "/images/" + image_files[i+1]
            #frame2 = cv.imread(image_filename)
            #frame2 = scale_image(frame2, 0.25)

            #frame = cv.addWeighted(frame, 0.5, frame2, 0.5, 0.0)
            
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            lower = np.array([0, 0, 0])
            upper = np.array([180, 60, 130])

            mask = cv.inRange(hsv_frame, lower, upper)
            hsv_frame[:,:,0] *= mask
            hsv_frame[:,:,1] *= mask
            hsv_frame[:,:,2] *= mask

            mask2 = hsv_frame[:,:,0] < (hsv_frame[:,:,1] * 0.9)
            hsv_frame[:,:,0] *= mask2
            hsv_frame[:,:,1] *= mask2
            hsv_frame[:,:,2] *= mask2

            edges = hsv_frame[:,:,0] + hsv_frame[:,:,1] + hsv_frame[:,:,2] > 0
            kernel = skimage.morphology.diamond(3).astype(np.uint8)
            edges = skimage.morphology.binary_closing(edges, kernel)

            lines = probabilistic_hough_line(edges, threshold=100, line_length=50, line_gap=2)

            for line in lines:
                p0, p1 = line
                color = (0, 0, 255)
                cv.line(hsv_frame, (p0[0], p0[1]), (p1[0], p1[1]), color)

            # Detect two radii
            hough_radii = np.arange(10, 40, 2)
            hough_res = hough_circle(edges, hough_radii)

            centers = []
            accums = []
            radii = []

            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract two circles
                peaks = peak_local_max(h, num_peaks=2)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius, radius])

            # Draw the most prominent 5 circles
            for idx in np.argsort(accums)[::-1][:5]:
                center_x, center_y = centers[idx]
                radius = radii[idx]
                cx, cy = circle_perimeter(center_y, center_x, radius)
                hsv_frame[cy, cx] = (255, 0, 0)

            cv.imshow('frame', hsv_frame)
            k = cv.waitKey(30) & 0xff

        cv.destroyAllWindows()

        annotation_timestamp = [None] * len(filename)

        return self.create_annotation_output(filename, frame_id, object_id, x_px, y_px, annotation_timestamp)

def main():
    video_dirname = "D:/Data/ZDO/224"
    tracker = InstrumentTracker()
    tracker.predict(video_dirname)

if __name__ == "__main__":
    main()