import DataMungingUtil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group into training, testing, and validation subfolders')
    parser.add_argument('--path', dest='path', type=str, required=True, help='path of root folder.')
    parser.add_argument('--width', dest='width', type=int, required=True, help='width of output images.')
    args = parser.parse_args()
    DataMungingUtil.resize_crop(args.path, args.width)
