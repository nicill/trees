import argparse
import os
import sys
import re
import cv2
import time
import numpy as np
from skimage.transform import resize as imresize
from torch.utils.data import DataLoader
from utils import color_codes, find_file
from datasets import Cropping2DDataset, CroppingDown2DDataset
from models import Unet2D
from metrics import hausdorf_distance, avg_euclidean_distance
from metrics import matched_percentage
from utils import list_from_mask

def checkGT(folder,siteName,sList):
    fileList=os.listdir(folder)
    #print(fileList)

    #go over each folder, if a GT file does not exist, create it
    if not(siteName+"GT.png" in fileList):
        print("no GT! "+siteName+"GT.png")
        #read ROI
        roiFile=folder+"/"+siteName+"ROI.jpg"
        roi=cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)
        if roi is None:raise Exception("NO ROI found in site "+roiFile+"")
        mask=roi.copy()
        mask[roi>0]=0 #now the mask is completely black

        #now go over the list of sites and add the code of the classes present
        for i in range(len(sList)):
            #print("label "+str(i))
            currentMaskFile=folder+"/"+siteName+sList[i]+".jpg"
            currentMask=cv2.imread(currentMaskFile,cv2.IMREAD_GRAYSCALE)
            if currentMask is None: print("species "+sList[i]+" not present in site "+siteName)
            else: mask[currentMask<150]=i #masks are black on white background


        # in the end, put everything outside thr ROI to background
        mask[roi>150]=0
        #print(np.unique(mask))
        cv2.imwrite(folder+"/"+siteName+"GT.png",mask)


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    parser.add_argument(
        '-d', '--mosaics-directory',
        dest='val_dir', # default='/home/mariano/Dropbox/DEM_Annotations',
        default='/home/mariano/Dropbox/280420',
        help='Directory containing the mosaics'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=20,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5,
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


def find_number(string):
    return int(''.join(filter(str.isdigit, string)))


def hsv_mosaics(mosaics, dems, cases):
    # Data loading (or preparation)
    options = parse_inputs()
    d_path = options['val_dir']

    hsv_mosaics = [
        cv2.cvtColor(mosaic, cv2.COLOR_BGR2HSV) for mosaic in mosaics
    ]
    hsv_mosaics = [
        np.stack([mosaic[..., 0], mosaic[..., 1], dem[..., 0]], -1)
        for mosaic, dem in zip(hsv_mosaics, dems)
    ]
    for mi, c_i in zip(hsv_mosaics, cases):
        cv2.imwrite(os.path.join(d_path, 'hsv_mosaic{:}.jpg'.format(c_i)), mi)


"""
Networks
"""


def train(cases, gt_names, roiNames, net_name, nClasses=47, verbose=1,resampleF=1):
    # Init
    print("\n\n\n\n STARTING TRAIN  ")
    options = parse_inputs()
    d_path = options['val_dir']
    c = color_codes()
    n_folds = len(gt_names)
    print("starting cases")
    print(cases)

    print("reading GT ")
    print(gt_names)

    y=[]
    counter=0
    for im in gt_names:
        image=cv2.imread(im,cv2.IMREAD_GRAYSCALE)
        if image is None: raise Exception("not read "+im)

        if resampleF!=1:
            image= np.argmax([cv2.resize((image==i).astype("uint8"), (int(image.shape[1]*resampleF),int(image.shape[0]*resampleF)),interpolation=cv2.INTER_LINEAR) for i in range(nClasses) ], axis=0)

            #resizedList=[]
            #for i in range(nClasses):
            #    toResize=(image==i).astype("uint8")
            #    resizedList.append(cv2.resize(toResize, (int(image.shape[0]*resampleF),int(image.shape[1]*resampleF)),interpolation=cv2.INTER_LINEAR))
            #image=np.argmax(resizedList)
            cv2.imwrite("gt"+str(counter)+".png",image)

        counter+=1
        #y.append((np.mean(image,axis=-1)< 50).astype(np.uint8))
        y.append(image.astype(np.uint8))

    #Print Unique values
    for yi in y: print(np.unique(yi))

    #y = [(np.mean(cv2.imread(im),axis=-1)< 50).astype(np.uint8)for im in gt_names]

    #print(y)

    if resampleF!=1:
        mosaics = []
        for c_i in cases:
            image=cv2.imread(c_i)
            mosaics.append(cv2.resize(image, (int(image.shape[1]*resampleF),int(image.shape[0]*resampleF)),interpolation=cv2.INTER_CUBIC))
        rois = []
        for c_i in roiNames:
            image=cv2.imread(c_i,cv2.IMREAD_GRAYSCALE)
            rois.append( (cv2.resize(image, (int(image.shape[1]*resampleF),int(image.shape[0]*resampleF)),
            interpolation=cv2.INTER_LINEAR) < 100).astype(np.uint8) )
    else:
        mosaics = [cv2.imread(c_i) for c_i in cases]
        rois = [(cv2.imread(c_i,cv2.IMREAD_GRAYSCALE) < 100).astype(np.uint8) for c_i in roiNames]

    originalSizes= []
    for c_i in cases:
        nowIm=cv2.imread(c_i)
        originalSizes.append((nowIm.shape[1],nowIm.shape[0]))

    #print(mosaics)
    x = [
         np.moveaxis(mosaic, -1, 0)
        for mosaic in mosaics
    ]

    mean_x = [np.mean(xi.reshape((len(xi), -1)), axis=-1) for xi in x]
    std_x = [np.std(xi.reshape((len(xi), -1)), axis=-1) for xi in x]

    norm_x = [
        (xi - meani.reshape((-1, 1, 1))) / stdi.reshape((-1, 1, 1))
        for xi, meani, stdi in zip(x, mean_x, std_x)
    ]

    print(
        '%s[%s] %sStarting cross-validation (leave-one-mosaic-out)'
        ' - %d mosaics%s' % (
            c['c'], time.strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    training_start = time.time()
    for i, case in enumerate(cases):
        if verbose > 0:
            print(
                '%s[%s]%s Starting training for mosaic %s %s(%d/%d)%s' %
                (
                    c['c'], time.strftime("%H:%M:%S"),
                    c['g'], case,
                    c['c'], i + 1, len(cases), c['nc']
                )
            )
        test_x = norm_x[i]

        train_y = y[:i] + y[i + 1:]
        train_roi = rois[:i] + rois[i + 1:]
        train_x = norm_x[:i] + norm_x[i + 1:]

        val_split = 0.1
        batch_size = 8
        patch_size = (256, 256)
        #patch_size = (64, 64)
        # overlap = (64, 64)
        overlap = (32, 32)
        num_workers = 1

        #model_name = '{:}.unc.mosaic{:}.mdl'.format(net_name, case)
        model_name = case[:-4]+"unc.mosaic"+net_name+".mdl"
        net = Unet2D(n_inputs=len(norm_x[0]),n_outputs=nClasses)

        print("MODEL NAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(model_name)

        try:
            net.load_model( model_name)
        except IOError:

            # Dataloader creation
            if verbose > 0:
                n_params = sum(
                    p.numel() for p in net.parameters() if p.requires_grad
                )
                print(
                    '%sStarting training with a Unet 2D%s (%d parameters)' %
                    (c['c'], c['nc'], n_params)
                )

            if val_split > 0:
                n_samples = len(train_x)

                n_t_samples = int(n_samples * (1 - val_split))

                d_train = train_x[:n_t_samples]
                d_val = train_x[n_t_samples:]

                l_train = train_y[:n_t_samples]
                l_val = train_y[n_t_samples:]

                r_train = train_roi[:n_t_samples]
                r_val = train_roi[n_t_samples:]

                print('Training dataset (with validation)')
                train_dataset = Cropping2DDataset(
                    d_train, l_train, r_train, patch_size=patch_size, overlap=overlap,
                    filtered=True
                )
#                 train_dataset = CroppingDown2DDataset(
#                     d_train, l_train, patch_size=patch_size, overlap=overlap,
#                     filtered=True
#                 )

                print('Validation dataset (with validation)')
                val_dataset = Cropping2DDataset(
                    d_val, l_val, r_val, patch_size=patch_size, overlap=overlap,
                    filtered=True
                )
#                 val_dataset = CroppingDown2DDataset(
#                     d_val, l_val, patch_size=patch_size, overlap=overlap,
#                     filtered=True
#                 )
            else:
                print('Training dataset')
                train_dataset = Cropping2DDataset(
                    train_x, train_y, train_roi, patch_size=patch_size, overlap=overlap,
                    filtered=True
                )

                print('Validation dataset')
                val_dataset = Cropping2DDataset(
                    train_x, train_y, train_roi, patch_size=patch_size, overlap=overlap
                )

            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )

            epochs = parse_inputs()['epochs']
            patience = parse_inputs()['patience']

            net.fit(
                train_dataloader,
                val_dataloader,
                epochs=epochs,
                patience=patience,
            )

            net.save_model( model_name)

        if verbose > 0:
            print(
                '%s[%s]%s Starting testing with mosaic %s %s(%d/%d)%s' %
                (
                    c['c'], time.strftime("%H:%M:%S"),
                    c['g'], case,
                    c['c'], i + 1, len(cases), c['nc']
                )
            )
        yi = net.test([test_x])
        pred_y = np.argmax(yi[0], axis=0)

        if resampleF!=1:
            cv2.imwrite(case[:-4]+"Result.png",cv2.resize(pred_y,originalSizes[i],interpolation=cv2.INTER_NEAREST).astype(np.uint8))
        else:
            cv2.imwrite(case[:-4]+"Result.png",
                (pred_y).astype(np.uint8)
            )

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - training_start)
        )
        print(
            '%sTraining finished%s (total time %s)\n' %
            (c['r'], c['nc'], time_str)
        )


def main():
    #S00 is the background class
    scalePercent=0.5
    maxSpeciesCode=11
    speciesList=["S"+'{:02d}'.format(i) for i in range(0,maxSpeciesCode+1)]
    speciesList.append("Other")

    print(speciesList)

    # Init
    options = parse_inputs()
    c = color_codes()

    # Data loading (or preparation)
    d_path = options['val_dir']
    siteFolders=sorted(os.listdir(d_path))

    #First, create ground truth files if they do not exist
    for x in siteFolders:checkGT(d_path+x,x,speciesList)

    gt_names = [d_path+"/"+x+"/"+x+"GT.png" for x in siteFolders ]
    print(gt_names)

    cases = [ d_path+x+"/"+x+".jpg" for x in siteFolders ]
    print("\n\n"+str(cases))

    rois = [ d_path+x+"/"+x+"ROI.jpg" for x in siteFolders ]
    print("\n\n")
    print(rois)

    ''' <Detection task> '''
    net_name = 'semantic-unet'
    train(cases, gt_names, rois, net_name, len(speciesList),1,scalePercent)


if __name__ == '__main__':
    main()
