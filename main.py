from arguments import parse_arguments
from camera import capture
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.lower_limit:
        MIN_FRAME_PROB = arguments.lower_limit

    # print(arguments.preview)
    capture(arguments.workdir, arguments.device, arguments.preview)
