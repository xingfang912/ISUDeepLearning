import DataMungingUtil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply local binary pattern filter to images and copy to new directory.')
    parser.add_argument('--path', dest='path', type=str, required=True, help='path of root folder.')
    args = parser.parse_args()
    DataMungingUtil.filter_lbp(args.path)
