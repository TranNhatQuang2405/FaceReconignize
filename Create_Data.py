import cv2
import numpy as np
from scipy.stats import itemfreq
from Detection import get_face, save_face
import pyautogui

width, height = pyautogui.size()

clicked = False


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


def resize_frame(image, COLOUR=[0, 0, 0]):
    h, w, layers = image.shape
    if h > height:
        ratio = height/h
        image = cv2.resize(
            image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if w > width:
        ratio = width/w
        image = cv2.resize(
            image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if h < height and w < width:
        hless = height/h
        wless = width/w
        if(hless < wless):
            image = cv2.resize(
                image, (int(image.shape[1] * hless), int(image.shape[0] * hless)))
        else:
            image = cv2.resize(
                image, (int(image.shape[1] * wless), int(image.shape[0] * wless)))
    h, w, layers = image.shape
    if h < height:
        df = height - h
        df /= 2
        image = cv2.copyMakeBorder(image, int(df), int(
            df), 0, 0, cv2.BORDER_CONSTANT, value=COLOUR)
    if w < width:
        df = width - w
        df /= 2
        image = cv2.copyMakeBorder(image, 0, 0, int(
            df), int(df), cv2.BORDER_CONSTANT, value=COLOUR)
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    return image


def main(folder, src):
    cameraCapture = cv2.VideoCapture(src)
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', onMouse)
    # Read and process frames in loop
    success, frame = cameraCapture.read()
    i = 0
    while success and not clicked:
        # i = i + 1
        success, frame = cameraCapture.read()
        frame = resize_frame(frame)
        result = get_face(frame)
        for j in range(len(result)):
            save_face(result[j], frame, folder, i)
            i = i + 1
        cv2.imshow('camera', frame)

        if(i > 200):
            break
        k = cv2.waitKey(1)

        if k == 27:
            break

    cv2.destroyAllWindows()
    cameraCapture.release()


if __name__ == "__main__":
    folder = 'HuuDuc'
    src = 0
    main(folder, src)
