import sys

def main(args):
    print("hi")


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("--input", default="Data/image-data", type=str, help="path to input image or folder of images")
    parser.add_argument("--output_folder", default="Results", type=str, help="folder to save the results in")
    
    parser.add_argument("--debugging", default=True, type=bool, help="whether to save images of the intermediate steps")
    parser.add_argument("--debugging_folder", default="debug", type=str, help="folder to save the debugging images in")

    return parser




if __name__ == "__main__":
    print(sys.argv[1])
    #args = get_args_parser().parse_args()    

    main(args=1)
    