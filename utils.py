import cv2
import numpy as np


def border_point(image, point, margin=100):
    """
    Function to check if a point (and its area) is in the border of the image.
    :param image: Binary image.
    :param point: Point to check.
    :param margin: Margin of error.
    :return:
    """
    top = np.array(image.shape)
    np.logical_or(point < margin, (top - point) < margin).any()

    return np.logical_or(point < margin, (top - point) < margin).any()


def list_from_binary(filename):
    """
    Function to take a binary image and output the center of masses of its
     connected regions.
    :param filename: Name of the file containing the binary image.
    :return:
    """
    # Open filename
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    point_list = list_from_image(im) if im is not None else []
    return point_list


def list_from_image(im):
    mask = cv2.threshold(255 - im, 40, 255, cv2.THRESH_BINARY)[1]
    return list_from_mask(mask)


def list_from_mask(mask):
    # Compute connected components
    n_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    # print(
    #     "crownSegmenterEvaluator, found {:d} {:d} points"
    #     " for file {:}".format(n_labels, len(centroids), filename)
    # )

    # print(" listFromBinary, found  {:d}".format(len(centroids)))
    # print(centroids)

    new_centroids = [c for c in centroids if not border_point(mask, c)]
    # print(" listFromBinary, refined  {:d}".format(len(new_centroids)))
    # print(new_centroids)

    return new_centroids[1:]