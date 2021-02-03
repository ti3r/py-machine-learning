import logging
from arguments import parse_arguments
from camera import capture

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.lower_limit:
        MIN_FRAME_PROB = arguments.lower_limit

    if arguments.trace_logs:
        logging.getLogger().setLevel(logging.DEBUG)

    # print(arguments.preview)
    capture(arguments.workdir, arguments.device, arguments.preview)
