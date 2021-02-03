import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default="0", help='Video device to use during capture', type=int,
                        required=False)

    parser.add_argument('-w', '--workdir', default="frames/", help='directory where to store frame images')

    parser.add_argument('-p', '--preview', default=False, action='store_true',
                        help='runs the application in headless mode. No display of preview images')

    parser.add_argument('-l', '--lower-limit', type=float,
                        help='probability lower limit to detect an object and capture a frame')

    parser.add_argument('-t', '--trace-logs', default=False, action='store_true',
                        help='Sets the logging level to trace (debug) in order to display more information in console')

    return parser.parse_args()

