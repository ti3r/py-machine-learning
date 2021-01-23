import cv2
import time
from detect import classify_image, prepare_interpreter
import logging
from datetime import datetime

PREVIEW_WINDOW_NAME = 'preview'
PREVIEW_WINDOW_SIZE = (960, 540)
FONT = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0, 1024)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def current_milli_time():
    return int(round(time.time() * 1000))


def check_for_exit():
    key = cv2.waitKey(5)
    if key == 27:  # exit on ESC
        return False

    if cv2.getWindowProperty(PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
        return False

    return True


def display_frame_in_prev_window(frame, text):
    cv2.putText(frame, text, bottomLeftCornerOfText, FONT, fontScale, fontColor, lineType)
    dis_frame = cv2.resize(frame, PREVIEW_WINDOW_SIZE)
    cv2.imshow(PREVIEW_WINDOW_NAME, dis_frame)


def write_frame_to_file(frame, results, workdir):
    label_id, prob = results[0]
    if prob > 0.5:
        cv2.imwrite('{workdir}/frame{num}.jpg'.format(workdir=workdir, num=current_milli_time())
                    , frame)


def display_results(frame, results, labels, preview):
    label_id, prob = results[0]
    label = ""
    if prob > 0.5:
        label = labels[label_id]
        print('label_id: {label}, label: {lab}, prob: {prob}, '.format(label=label_id, prob=prob, lab=label))

    if preview:
        display_frame_in_prev_window(frame, label)


def log_message(is_preview):
    logging.debug("Frame captured on ${datetime.now()}")
    if not is_preview:
        logging.info("Frame...")


def capture(workdir, device=0, preview=True):
    vc = cv2.VideoCapture(device)

    if preview:
        cv2.namedWindow(PREVIEW_WINDOW_NAME)

    rval, frame = vc.read() if vc.isOpened() else [False, None]  # try to get the first frame
    interpreter, labels = prepare_interpreter()

    while rval:
        rval, frame = vc.read()
        ml_frame = cv2.resize(frame, (224, 224))
        results = classify_image(interpreter, ml_frame)
        write_frame_to_file(frame, results, workdir)
        display_results(frame, results, labels, preview)
        rval = check_for_exit()
        log_message(preview)

    cv2.destroyWindow(PREVIEW_WINDOW_NAME)
