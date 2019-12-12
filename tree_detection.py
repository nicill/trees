import argparse
import os
import re
from time import strftime
import numpy as np
from data_manipulation.utils import color_codes, find_file
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


def train_test_net(net_name, verbose=1):
    """

    :param net_name:
    :return:
    """
    # Init
    c = color_codes()
    options = parse_inputs()

    # Data loading (or preparation)
    d_path = options['val_dir']
    gt_names = sorted(list(filter(
        lambda x: not os.path.isdir(x) and re.search(options['lab_tag'], x),
        os.listdir(d_path)
    )))
    n_folds = len(gt_names)
    cases = [re.search(r'(\d+)', r).group() for r in gt_names]

    print(
        '%s[%s] %sStarting cross-validation (leave-one-mosaic-out)'
        ' - %d mosaics%s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    for i, case in enumerate(cases):
        if verbose > 0:
            print(
                '%s[%s]%s Starting training for mosaic %s %s(%d/%d)%s' %
                (
                    c['c'], strftime("%H:%M:%S"),
                    c['g'], case,
                    c['c'], i + 1, len(cases), c['nc']
                )
            )

        test_gt_name = gt_names[i]
        test_dem_name = 'DEM{:}.jpg'.format(case)
        test_mosaic_name = 'mosaic{:}.jpg'.format(case)

        train_gt_names = gt_names[:i] + gt_names[i + 1:]
        train_dem_names = [
            'DEM{:}.jpg'.format(c_i)
            for c_i in cases[:i] + cases[i + 1:]
        ]
        train_mosaic_names = [
            'mosaic{:}.jpg'.format(c_i)
            for c_i in cases[:i] + cases[i + 1:]
        ]
        print(
            find_file(test_gt_name, d_path),
            find_file(test_dem_name, d_path),
            find_file(test_mosaic_name, d_path)
        )


def main():
    # Init
    c = color_codes()

    print(
        '%s[%s] %s<Tree detection pipeline>%s' % (
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    ''' <Detection task> '''
    net_name = 'tree-detection.unet'

    train_test_net(net_name)


if __name__ == '__main__':
    main()