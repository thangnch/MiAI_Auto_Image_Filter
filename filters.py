import numpy as np
import cv2
import random
import glob
import math


def verify_alpha_channel(frame):
    try:
        frame.shape[3]  # looking for the alpha channel
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame


def apply_hue_saturation(frame, alpha, beta):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s.fill(199)
    v.fill(255)
    hsv_image = cv2.merge([h, s, v])

    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    frame = verify_alpha_channel(frame)
    out = verify_alpha_channel(out)
    cv2.addWeighted(out, 0.25, frame, 1.0, .23, frame)
    return frame


def apply_color_overlay(frame, intensity=0.5, blue=0, green=0, red=0):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    sepia_bgra = (blue, green, red, 1)
    overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    return frame


def apply_sepia(frame, intensity=0.5):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    sepia_bgra = (20, 66, 112, 1)
    overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    return frame


def alpha_blend(frame_1, frame_2, mask):
    alpha = mask / 255.0
    blended = cv2.convertScaleAbs(frame_1 * (1 - alpha) + frame_2 * alpha)
    return blended


def apply_circle_focus_blur(frame, intensity=0.2):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    y = int(frame_h / 2)
    x = int(frame_w / 2)

    mask = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    cv2.circle(mask, (x, y), int(y / 2), (255, 255, 255), -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)

    blured = cv2.GaussianBlur(frame, (21, 21), 11)
    blended = alpha_blend(frame, blured, 255 - mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame


def portrait_mode(frame):
    cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    blured = cv2.GaussianBlur(frame, (21, 21), 11)
    blended = alpha_blend(frame, blured, mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame


def apply_invert(frame):
    return cv2.bitwise_not(frame)

