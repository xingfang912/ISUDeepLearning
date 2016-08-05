from ISUDeepLearning.DataMungingUtil import equalize_num_samples
import argparse

# run this after "group_into_class_folders.py" but before "partition.py"
# Removes classes that have fewer than n samples, and reduces the number of samples that have more than n.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group into training, testing, and validation subfolders')
    parser.add_argument('--path', dest='path', type=str, required=True, help='path of root folder.')
    parser.add_argument('--n', dest='n', type=int, required=True, help='number of samples per class')
    args = parser.parse_args()
    equalize_num_samples(args.path, args.n)
