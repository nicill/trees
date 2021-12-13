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
from datasets import Cropping2DDataset
from models import Unet2D
from metrics import hausdorf_distance, avg_euclidean_distance
from metrics import matched_percentage
from utils import list_from_mask

def toSingleList(aListOfLists,excludedIndex):
    returnList=[]
    for i in range(len(aListOfLists)):
        if i!=excludedIndex:
            for x in aListOfLists[i]:returnList.append(x)
    return returnList

def toListOfLists(aList,indexDict):
    returnList=[]
    for k,v in indexDict.items():
        returnList.append([aList[i] for i in v])
    return returnList

def tellApartSitesFromMosaics(folders):
    myDict={"S"+x.split("S")[1]:[] for x in folders}
    i=0
    for x in folders:
        myDict["S"+x.split("S")[1]].append(i)
        i+=1

    #print(myDict)
    return myDict

def fuseSpeciesList(tagFile):# Read information on how to fuse species and return it as a list and dictionary
    speciesList=["S00"]
    speciesDict={}
    inverseSpeciesDict={}
    speciesDict["S00"]=0
    inverseSpeciesDict[0]=1
    with open(tagFile) as fp:
        line = fp.readline()
        while line:
            currentLabel=line.split(" ")[0]
            nextLabel=line.split(" ")[1].strip()
            if nextLabel not in inverseSpeciesDict:inverseSpeciesDict[nextLabel]=1
            else:inverseSpeciesDict[nextLabel]+=1
            speciesList.append(currentLabel)
            speciesDict[currentLabel]=int(nextLabel)
            line = fp.readline()

    return speciesList,speciesDict,len(inverseSpeciesDict)

def checkGT(folder,siteName,sList,sDict):
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
            else: mask[currentMask<150]=sDict[sList[i]] #masks are black on white background


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
        '-labFus', '--label-fusion',
        dest='labTab',
        default=None,
        help='text file with the code for classes to be fused'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=20,
        help='Number of epochs'
    )
    parser.add_argument(
        '-aug', '--augment',
        dest='augment',
        type=int,  default=0,
        help='augmentations per image'
    )
    parser.add_argument(
        '-dec', '--decrease',
        dest='decrease',
        type=int,  default=0,
        help='decreasing percentage per 100 images'
    )
    parser.add_argument(
        '-th', '--threshold',
        dest='threshold',
        type=int,  default=0,
        help='minumin threshold over 100 to consider a pixel decided'
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
    parser.add_argument(
        '-u2', '--unet2',
        dest='u2', default=False,
        help='whether or not we use a second unet'
    )

    options = vars(parser.parse_args())

    return options


def find_number(string):
    return int(''.join(filter(str.isdigit, string)))

def invertIndicesDict(sitesIndicesDict):
    returnDict={}
    for k,v in sitesIndicesDict.items():
        for x in v:
            if x not in returnDict:returnDict[x]=k
    return returnDict

"""
Networks
"""
def train(cases, gt_names, roiNames, net_name, dictSitesMosaics, nClasses=47, verbose=1,resampleF=1):
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

#TODO DO BETTER!!!!!!!
    #Unet 1
    codedImportant=[4,5,6] #actively increase
    codedUnImportant=[0,1,2,3] #actively decrease
    codedIgnore=[]
    #Unet2
    useSecondNet=bool(options["u2"])
    if useSecondNet:
        important2=[4,5,6]
        unimportant2=[]
        ignore2=[0,1,2,3]

    # Add a unet2 parameter, if it is present, define new training/validation dataset with the important2, unimportant2,ignore2 lists.
    # define and train a second unet based on those dataloaders,
    # for the pixels where the first unet is undecided, call the second unet

    augment=parse_inputs()['augment']
    decreaseRead=parse_inputs()['decrease']
    decrease=decreaseRead/100.

    y=[]
    counter=0
    for im in gt_names:
        image=cv2.imread(im,cv2.IMREAD_GRAYSCALE)
        if image is None: raise Exception("not read "+im)

        if resampleF!=1:
            image= np.argmax([cv2.resize((image==i).astype("uint8"), (int(image.shape[1]*resampleF),int(image.shape[0]*resampleF)),interpolation=cv2.INTER_LINEAR) for i in range(nClasses) ], axis=0)

        counter+=1
        y.append(image.astype(np.uint8))

    #Print Unique values
    for yi in y: print(np.unique(yi))

    if resampleF!=1:
        mosaics = []
        counter=0
        for c_i in cases:
            image=cv2.imread(c_i)
            mosaics.append(cv2.resize(image, (int(image.shape[1]*resampleF),int(image.shape[0]*resampleF)),interpolation=cv2.INTER_CUBIC))
            #cv2.imwrite("mos"+str(counter)+".png",mosaics[-1])
            counter+=1

        rois = []
        counter=0
        for c_i in roiNames:
            image=cv2.imread(c_i,cv2.IMREAD_GRAYSCALE)
            rois.append( (cv2.resize(image, (int(image.shape[1]*resampleF),int(image.shape[0]*resampleF)),
            interpolation=cv2.INTER_LINEAR) < 100).astype(np.uint8) )
            #cv2.imwrite("ROI"+str(counter)+".png",rois[-1])
            counter+=1
    else:
        mosaics = [cv2.imread(c_i) for c_i in cases]
        rois = [(cv2.imread(c_i,cv2.IMREAD_GRAYSCALE) < 100).astype(np.uint8) for c_i in roiNames]

    originalSizes= []
    for c_i in cases:
        nowIm=cv2.imread(c_i)
        originalSizes.append((nowIm.shape[1],nowIm.shape[0]))

    #print(mosaics)
    #print("LETs make lists of lists with "+str(dictSitesMosaics))

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

    # create also a list of testIndices divided by sites
    indices=[i for i in range(len(x))]
    indices=toListOfLists(indices,dictSitesMosaics)
    #print("indices starting")
    #print(indices)
    x=toListOfLists(x,dictSitesMosaics)
    mean_x=toListOfLists(mean_x,dictSitesMosaics)
    std_x=toListOfLists(std_x,dictSitesMosaics)
    norm_x=toListOfLists(norm_x,dictSitesMosaics)
    y=toListOfLists(y,dictSitesMosaics)
    rois=toListOfLists(rois,dictSitesMosaics)
    cases=toListOfLists(cases,dictSitesMosaics)

    print(
        '%s[%s] %sStarting cross-validation (leave-one-mosaic-out)'
        ' - %d mosaics%s' % (
            c['c'], time.strftime("%H:%M:%S"), c['g'], n_folds, c['nc']
        )
    )
    training_start = time.time()
    print(cases)
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

        test_x = norm_x[i]#test_x is now a list!!!!!!!!!!!!!!!!!!!!!
        #create an inverse dictionary!
        #dictMosaicsSites=invertIndicesDict(dictSitesMosaics)
        train_y = toSingleList(y,i)
        train_roi = toSingleList(rois,i)
        train_x = toSingleList(norm_x,i)
        print("indices train "+str(toSingleList(indices,i)))

        val_split = 0.1
        batch_size = 8
        patch_size = (256, 256)
        #patch_size = (64, 64)
        # overlap = (64, 64)
        overlap = (32, 32)
        num_workers = 1

        # We only store one model in case there are multiple flights
        model_name = case[0][:-4]+"unc.mosaic"+net_name+"augm"+str(augment)+"decrease"+str(decreaseRead)+".mdl"
        print("MODEL NAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(model_name)
        if useSecondNet:model_name2 = case[0][:-4]+"unc.mosaic"+net_name+"augm"+str(augment)+"decrease"+str(decreaseRead)+"SECOND.mdl"

        #net = Unet2D(n_inputs=len(norm_x[0]),n_outputs=nClasses)
        # NO ENTENC GAIRE PERQUE FUNCIONA!!!!!
        net = Unet2D(n_inputs=len(norm_x[0][0]),n_outputs=nClasses)
        if useSecondNet:
            net2 = Unet2D(n_inputs=len(norm_x[0][0]),n_outputs=nClasses)
            print("2nd MODEL NAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(model_name2)
        else: print("NOT USING SECOND UNET")


        try:
            net.load_model( model_name)
            if useSecondNet:net.load_model( model_name2)
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

                print('Training datasets (with validation)')
                train_dataset = Cropping2DDataset(
                    d_train, l_train, r_train, numLabels=nClasses,important=codedImportant, unimportant=codedUnImportant, ignore=codedIgnore,augment=augment,decrease=decrease, patch_size=patch_size, overlap=overlap
                )


                print('Validation datasets (with validation)')
                val_dataset = Cropping2DDataset(
                    d_val, l_val, r_val, numLabels=nClasses,important=codedImportant, unimportant=codedUnImportant, ignore=codedIgnore,augment=augment,decrease=decrease, patch_size=patch_size, overlap=overlap
                )

                if useSecondNet:
                    train_dataset2 = Cropping2DDataset(
                    d_train, l_train, r_train, numLabels=nClasses,important=important2, unimportant=unimportant2, ignore=ignore2,augment=augment,decrease=decrease, patch_size=patch_size, overlap=overlap
                )
                    val_dataset2 = Cropping2DDataset(
                    d_val, l_val, r_val, numLabels=nClasses,important=important2, unimportant=unimportant2, ignore=ignore2,augment=augment,decrease=decrease, patch_size=patch_size, overlap=overlap
                )


            else:
                raise Exception("NOT DOING THIS!")
                """
                print('Training dataset')
                train_dataset = Cropping2DDataset(
                    train_x, train_y, train_roi, numLabels=nClasses,important=codedImportant, unimportant=codedUnImportant,augment=augment,decrease=decrease,  patch_size=patch_size, overlap=overlap
                )

                print('Validation dataset')
                val_dataset = Cropping2DDataset(
                    train_x, train_y, train_roi, numLabels=nClasses,important=codedImportant, unimportant=codedUnImportant,augment=augment,decrease=decrease, patch_size=patch_size, overlap=overlap
                )
                """

            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )

            if useSecondNet:
                train_dataloader2 = DataLoader(
                train_dataset2, batch_size, True, num_workers=num_workers
            )
                val_dataloader2 = DataLoader(
                val_dataset2, batch_size, num_workers=num_workers
            )


            epochs = parse_inputs()['epochs']
            patience = parse_inputs()['patience']

            net.fit(
                train_dataloader,
                val_dataloader,
                epochs=epochs,
                patience=patience,
            )

            if useSecondNet:
                    print("TRAINING SECOND MODEL!!!!!!!!!!!!")
                    net2.fit(
                    train_dataloader2,
                    val_dataloader2,
                    epochs=epochs,
                    patience=patience,
            )


            net.save_model( model_name)
            if useSecondNet:net2.save_model( model_name2)

        if verbose > 0:
            print(
                '%s[%s]%s Starting testing with mosaic %s %s(%d/%d)%s' %
                (
                    c['c'], time.strftime("%H:%M:%S"),
                    c['g'], case,
                    c['c'], i + 1, len(cases), c['nc']
                )
            )

        for ind in range(len(test_x)):
            yi = net.test([test_x[ind]])
            pred_y = np.argmax(yi[0], axis=0)
            heatMap_y = np.max(yi[0], axis=0)

            #now exclude classes with probability under the thershold
            thRead=parse_inputs()['threshold']
            probTH=thRead/100.
            pred_y[heatMap_y<probTH]=255

            # now reclassify usingn second Unet
            if useSecondNet:
                yi2 = net2.test([test_x[ind]])
                pred_y2 = np.argmax(yi2[0], axis=0)
                #heatMap_y = np.max(yi[0], axis=0)
                pred_y[heatMap_y==255]=0
                pred_y2[heatMap_y!=255]=0
                pred_y=pred_y+pred_y2

            if resampleF!=1:
                cv2.imwrite(case[ind][:-4]+"augm"+str(augment)+"decrease"+str(decreaseRead)+"ResultTH"+str(thRead)+".png",cv2.resize(pred_y,originalSizes[i],interpolation=cv2.INTER_NEAREST).astype(np.uint8))
                cv2.imwrite(case[ind][:-4]+"augm"+str(augment)+"decrease"+str(decreaseRead)+"ResultTH"+str(thRead)+"SMALL.png",
                    (pred_y).astype(np.uint8)
                )

            else:
                cv2.imwrite(case[ind][:-4]+"augm"+str(augment)+"decrease"+str(decreaseRead)+"ResultTH"+str(thRead)+".png",
                    (pred_y).astype(np.uint8)
                )
            print("Results FILE!!!!!!!!!!!!!!!!!!!!! "+str(case[ind][:-4]+"augm"+str(augment)+"decrease"+str(decreaseRead)+"ResultTH"+str(thRead)+".png"))

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

    # Init
    options = parse_inputs()
    c = color_codes()

    #S00 is the background class
    scalePercent=1
    maxSpeciesCode=46
    tagFile=options['labTab']
    speciesDict={}
    if tagFile is None:
        speciesList=["S"+'{:02d}'.format(i) for i in range(0,maxSpeciesCode+1)]
        for sp in range(len(speciesList)):speciesDict[speciesList[sp]]=sp
    else:
        speciesList,speciesDict,numClasses=fuseSpeciesList(tagFile)

    print(speciesList)
    print(speciesDict)
    print("Number of different classes "+str(numClasses))

    # Data loading (or preparation)
    d_path = options['val_dir']
    siteFolders=sorted(os.listdir(d_path))

    #First, create ground truth files if they do not exist
    for x in siteFolders:checkGT(d_path+x,x,speciesList,speciesDict)

    #also return a dictionary of sites and their mosaics (indices in cases?)

    gt_names = [d_path+"/"+x+"/"+x+"GT.png" for x in siteFolders ]
    print(gt_names)

    cases = [ d_path+x+"/"+x+".jpg" for x in siteFolders ]
    print("\n\n"+str(cases))

    rois = [ d_path+x+"/"+x+"ROI.jpg" for x in siteFolders ]
    print("\n\n")
    print(rois)

    dictSitesMosaics=tellApartSitesFromMosaics(siteFolders)
    print(dictSitesMosaics)

    ''' <Detection task> '''
    net_name = 'semantic-unet'
    train(cases, gt_names, rois, net_name,dictSitesMosaics , numClasses,1,scalePercent)


if __name__ == '__main__':
    main()
