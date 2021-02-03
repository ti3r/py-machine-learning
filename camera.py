import cv2
import time
from detect import classify_image, prepare_interpreter
import logging
from datetime import datetime

MIN_FRAME_PROB = 0.75
PREVIEW_WINDOW_NAME = 'preview'
PREVIEW_WINDOW_SIZE = (960, 540)
FONT = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0, 1024)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def current_milli_time():
    return int(round(time.time() * 1000))


def check_for_exit(preview):
    key = cv2.waitKey(5)
    if key == 27:  # exit on ESC
        return False

    if preview and cv2.getWindowProperty(PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
        return False

    return True


def display_frame_in_prev_window(frame, text):
    cv2.putText(frame, text, bottomLeftCornerOfText, FONT, fontScale, fontColor, lineType)
    dis_frame = cv2.resize(frame, PREVIEW_WINDOW_SIZE)
    cv2.imshow(PREVIEW_WINDOW_NAME, dis_frame)


def write_frame_to_file(frame, results, workdir):
    for result in results:
        label, prob = result
        if prob > MIN_FRAME_PROB:
            num = current_milli_time()
            cv2.imwrite('{workdir}/frame{num}-{lab}.jpg'.format(workdir=workdir, num=num, lab=label), frame)


def display_results(frame, results, preview):
    for result in results:
        label, prob = result
        logging.info('label: {lab}, prob: {prob}, '.format(lab=label, prob=prob))

    if preview:
        display_frame_in_prev_window(frame, '')


def log_message(is_preview):
    logging.debug(f'Frame captured on {datetime.now()}')
    if not is_preview:
        logging.info("Frame...")


def process_image(interpreter, frame, ml_frame, labels, workdir, preview):
    results = classify_image(interpreter, ml_frame, labels, MIN_FRAME_PROB, top_k=5)
    write_frame_to_file(frame, results, workdir)
    log_message(preview)
    display_results(frame, results, preview)


def capture(workdir, device=0, preview=True):
    logging.info(f'Initializing capture device: {device}')

    vc = cv2.VideoCapture(device)

    if preview:
        cv2.namedWindow(PREVIEW_WINDOW_NAME)

    rval, frame = vc.read() if vc.isOpened() else [False, None]  # try to get the first frame
    interpreter, labels, dimentions = prepare_interpreter()
    logging.debug("Tensorflow interpreter ready...")

    while rval:
        rval, frame = vc.read()
        ml_frame = cv2.resize(frame, (dimentions[0], dimentions[1]))
        process_image(interpreter, frame, ml_frame, labels, workdir, preview)
        rval = check_for_exit(preview)

    cv2.destroyWindow(PREVIEW_WINDOW_NAME)
