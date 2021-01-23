import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default="0", help='Video device to use during capture', type=int,
                        required=False)

    parser.add_argument('-w', '--workdir', default="frames/", help='directory where to store frame images',
                        required=False)

    parser.add_argument('-hl', '--headless', default="True",
                        help='runs the application in headless mode. No display of preview images')
    return parser.parse_args()

