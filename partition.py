import DataMungingUtil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group into training, testing, and validation subfolders')
    parser.add_argument('--path', dest='path', type=str, required=True, help='path of root folder.')
    parser.add_argument('--training', dest='training', type=int, required=True, help='number of training samples per class')
    parser.add_argument('--validation', dest='validation', type=int, required=True, help='number of validation samples per class')
    parser.add_argument('--testing', dest='testing', type=int, required=True, help='number of testing samples per class')
    args = parser.parse_args()
    DataMungingUtil.partition(args.path, args.training, args.validation, args.testing)
