from ISUDeepLearning.DataMungingUtil import group_into_class_folders
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group into class folders')
    parser.add_argument('--path', dest='path', type=str, required=True, help='path of root folder.')
    args = parser.parse_args()
    group_into_class_folders(args.path)
