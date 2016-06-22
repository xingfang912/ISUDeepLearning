import DataMungingUtil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format images in class subfolders for caffe. Generate lmdb file.')
    parser.add_argument('--width', dest='width', type=int, required=True, help='width of output images.')
    args = parser.parse_args()
    DataMungingUtil.format_4_caffe(target_width=args.width)
