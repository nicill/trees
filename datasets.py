import itertools
from skimage.transform import resize as imresize
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from imgaug import augmenters as iaa


def augment(image,code,verbose=True):
    #outputFile="./fuio.jpg"
    #print("shape!!! "+str(image.shape))
    if code==0:
        if verbose: print("Doing Data augmentation 0 (H fip) to image ")
        image_aug = iaa.Rot90(1)(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==1:
        if verbose: print("Doing Data augmentation 1 (V flip) to image ")
        image_aug = iaa.Flipud(1.0)(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==2:
        if verbose: print("Doing Data augmentation 2 (Gaussian Blur) to image ")
        image_aug = iaa.GaussianBlur(sigma=(0, 0.5))(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==3:
        if verbose: print("Doing Data augmentation 3 (rotation) to image ")
        angle=random.randint(0,45)
        rotate = iaa.Affine(rotate=(-angle, angle))
        image_aug = rotate(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==4:
        if verbose: print("Doing Data augmentation 4 (elastic) to image ")
        image_aug = iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1)(image=image)
        #cv2.imwrite(outputFile,image_aug)
    elif code==5:
        if verbose: print("Doing Data augmentation 5 (contrast) to image ")
        image_aug=iaa.LinearContrast((0.75, 1.5))(image=image)
        #cv2.imwrite(outputFile,image_aug)
    else:
        raise Exception("datasets.py Data augmentation technique not recognised ")
    return np.moveaxis(image_aug,-1,0)


def centers_to_slice(voxels, patch_half):
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        [
            np.concatenate([np.arange(*t), [t[1]]])
            for t in zip(min_bb_i, max_bb_i, steps)
        ] for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


class Cropping2DDataset(Dataset):
    def __init__(
            self,
            data, labels, rois, numLabels,important=[], unimportant=[],augment=0,decrease=0,patch_size=32, overlap=16
    ):
        # Init
        self.data = data
        self.labels = labels
        self.rois = rois
        data_shape = self.data[0].shape

        # create a list of pixels by label
        self.labelStats=[]
        for i in range(numLabels):self.labelStats.append([i,0,0])
        #print(self.labelStats)

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size
        self.overlap = overlap

        slices = get_slices(
            self.rois, self.patch_size, self.overlap
        )

        # Filter out patches completely outside the ROI
        self.patch_slices = [
            (s, i) for i, (roi, slices_i) in enumerate(zip(self.rois, slices))
            for s in slices_i if np.sum(roi[s]) > 0
        ]

        # Now traverse all real patches and count how many pixels of each class there are inside the ROI
        for i in range(len(self.patch_slices)):
            input,targ=self.accessRaw(i)
            im=targ
            roi=input[1]
            im[roi>150]=-1
            # now count each pixel of each class inside the patch and the ROI
            for i in range(numLabels):self.labelStats[i][1]+=np.sum(im==i)

        #compute percentages
        totalRelevantPixels=0
        for c,pix,zero in self.labelStats: totalRelevantPixels+=pix
        for x in self.labelStats: x[2]=100*x[1]/totalRelevantPixels

        #now we know how many pixels are in the relevant classes
        #now create a list of indices and
        # now perform data augmentation, go over the whole dataset
        # ignore patches with no labels!
        # for the patches from interesting classes, make a note to later make augmentFactor copies
        self.AugmDict={}
        self.realCount=0
        #uncount=0
        for i in range(len(self.patch_slices)):
            im,targ=self.accessRaw(i)
            #if 1 in targ:# at least some label present
            if patchWithClasses(targ):# at least some label present
                if (not uninterestingPatch(targ,self.uninteresting)) or random.randint(0,99)>self.downSampleUninterestingPercentage:#random draw
                    self.AugmDict[self.realCount]=i
                    self.realCount+=1
                    if uninterestingPatch(targ,self.uninteresting):self.countUninteresting+=1

                #else:
                #    print("ignoring unimportant patch "+str(uncount))
                #    uncount+=1

        print("now go to augment, real "+str(self.realCount)+" there were "+str(self.max_slice[-1])+" augment factor "+str(augmentFactor))

        if augment:
            augmCount=self.realCount
            for i in range(self.realCount):
                im,targDouble=self[i]
                targ=targDouble[0]
                # targ is a list of probabilities if the patch belongs to the class at that position
                patchToAugment=False
                superInterestingPatch=True
                labelsInPatch=0
                j=0
                while j<self.numLabels:
                    #print("targ "+str(targ[j]))
                    #if targ[j]==1 and j in self.interesting:
                    if targ[j]!=0 and j in self.interesting:
                       patchToAugment=True
                       labelsInPatch+=1
                    #elif targ[j]==1:
                    elif targ[j]!=0:
                        labelsInPatch+=1
                        superInterestingPatch=False
                    j+=1

                #if superInterestingPatch, double the augment factor
                if superInterestingPatch and (labelsInPatch!=0):
                    self.countInteresting+=1
                    #print("super interesting!!! "+str(targ)+" labels in patch "+str(labelsInPatch)+" "+str((labelsInPatch!=0)) )
                    for k in range(2*augmentFactor):
                        self.countInteresting+=1
                        self.AugmDict[augmCount]=i
                        augmCount+=1
                elif patchToAugment: #if not superinteresting but contains interesting, augment with the normal augment factor
                    self.countInteresting+=1
                    for k in range(augmentFactor):
                        self.countInteresting+=1
                        self.AugmDict[augmCount]=i
                        augmCount+=1


        print("FINISHED CREATING DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(self.AugmDict)
        self.len=len(self.AugmDict)
        print("length was "+str(self.realCount)+" and will become "+str(self.len)+" interesting patches made "+str(100*self.countInteresting/self.realCount)+" uninteresting patches left "+str(100*self.countUninteresting/self.realCount))






    def accessRaw(self, index):
        # We select the case
        slice_i, case_idx = self.patch_slices[index]

        # We get the slice indexes
        none_slice = (slice(None, None),)

        inputs = (
            self.data[case_idx][none_slice + slice_i].astype(np.float32),
            np.expand_dims(
                self.rois[case_idx][slice_i].astype(np.uint8), axis=0
            )
        )

        target = np.expand_dims(
            self.labels[case_idx][slice_i].astype(np.uint8), axis=0
        )

        # target_labs = bwlabeln(target.astype(np.bool))
        # tops = len(np.unique(target_labs[target.astype(np.bool)]))

        return inputs, target

    def __getitem__(self, index):
        # We select the case
        slice_i, case_idx = self.patch_slices[index]

        # We get the slice indexes
        none_slice = (slice(None, None),)

        inputs = (
            self.data[case_idx][none_slice + slice_i].astype(np.float32),
            np.expand_dims(
                self.rois[case_idx][slice_i].astype(np.uint8), axis=0
            )
        )

        target = np.expand_dims(
            self.labels[case_idx][slice_i].astype(np.uint8), axis=0
        )

        # target_labs = bwlabeln(target.astype(np.bool))
        # tops = len(np.unique(target_labs[target.astype(np.bool)]))

        return inputs, target

    def __len__(self):
        return len(self.patch_slices)
