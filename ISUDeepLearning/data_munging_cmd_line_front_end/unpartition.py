from ISUDeepLearning.DataMungingUtil import unpartition
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ungroup from training, testing, and validation subfolders')
    parser.add_argument('--path', dest='path', type=str, required=True, help='path of root folder.')
    args = parser.parse_args()
    unpartition(args.path)
