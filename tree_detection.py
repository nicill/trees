import argparse
import os
import re
from time import strftime
import numpy as np
from data_manipulation.utils import color_codes
from matplotlib import image as mpimage


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    parser.add_argument(
        '-v', '--mosaics-directory',
        dest='val_dir', default='/home/mariano/Dropbox/DEM_Annotations',
        help='Directory containing the mosaics'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=2,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-B', '--batch-size',
        dest='batch_size',
        type=int, default=32,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-t', '--patch-size',
        dest='patch_size',
        type=int, default=128,
        help='Patch size'
    )
    parser.add_argument(
        '-l', '--labels-tag',
        dest='lab_tag', default='top',
        help='Tag to be found on all the ground truth filenames'
    )

    options = vars(parser.parse_args())

    return options


def main():
    # Init
    c = color_codes()

    print(
        '%s[%s] %s<Tree detection pipeline>%s' % (
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    ''' <Detection task> '''
    options = parse_inputs()
    d_path = options['val_dir']
    gt_names = list(filter(
        lambda x: not os.path.isdir(x) and re.search(options['lab_tag'], x),
        os.listdir(d_path)
    ))
    n_folds = len(gt_names)
    print(
        '%s[%s] %sStarting cross-validation (leave-one-mosaic-out)'
        ' - %d mosaics%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )

    net_name = 'brats2019-gen'

    train_test_seg(net_name, n_folds)


if __name__ == '__main__':
    main()