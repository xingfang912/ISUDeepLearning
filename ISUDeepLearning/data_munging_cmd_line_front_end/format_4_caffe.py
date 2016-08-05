from ISUDeepLearning.DataMungingUtil import format_4_caffe
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format images in class subfolders for caffe. Generate lmdb file.')
    parser.add_argument('--width', dest='width', type=int, required=True, help='width of output images.')
    parser.add_argument('--path', dest='path', type=str, required=True, help='root path where image class folders are stored.')
    args = parser.parse_args()
    format_4_caffe(path=args.path, target_width=args.width)
