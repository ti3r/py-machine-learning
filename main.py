from arguments import parse_arguments
from camera import capture


def run():
    print("Hello World!!!")


if __name__ == "__main__":
    arguments = parse_arguments()
    # print(arguments.image)
    capture(arguments.workdir, arguments.device)
