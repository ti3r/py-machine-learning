from arguments import parse_arguments
from camera import capture
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


if __name__ == "__main__":
    arguments = parse_arguments()
    # print(arguments.image)
    capture(arguments.workdir, arguments.device, not arguments.headless)
