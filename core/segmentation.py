import cv2
import numpy as np

def preprocess(gray, d=9, sigma=30):
    return cv2.bilateralFilter(gray, d, sigma, sigma)

def otsu_threshold(gray):
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def morph_cleanup(binary, close_k=3, open_k=5):
    sel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    sel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, sel1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, sel2)
    return opening

def remove_small_objects(binary, min_area=500):
    final = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(final, [c], -1, 255, -1)
            kept += 1

    return final, kept