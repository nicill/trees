import sys, getopt
import numpy as np
from PIL import Image
from PIL import ImageOps
import warnings
import cv2

#return the percentage of black pixels in the ground truth covered by the predicted mask
def coveredPerc(gtMask,predictedMask):
    total=np.sum(gtMask == 255)
    auxMask=gtMask.copy()
    #first, compute initial GT pixels
    #print("total gt pixels "+str(total))
    auxMask[predictedMask==255]=0
    covered=np.sum(auxMask == 255)
    #print("total covered gt pixels "+str(covered))

    return 1-(covered/total)

def FPPerc(gtMask,predictedMask):
    #first, compute initial GT pixels
    total=np.sum(gtMask == 255)
    auxMask=predictedMask.copy()
    auxMask[gtMask==255]=0
    FP=np.sum(auxMask == 255)
    return FP/total

def equalValue(mask1,mask2, ROI=None):# return the percentage of pixels with the same value

    try:
        totalPixels=np.sum(ROI!=0)
        #print("total Pixels in ROI "+str(totalPixels))
    except:
        raise Exception("NO ROI")
        #totalPixels=mask1.shape[0]*mask1.shape[1]

    im=mask1-mask2 # subtract masks to find out equal pixels

    im[ROI==0]=255
    im[im!=0]=255
    #cv2.imwrite("sampulla.png",im)

    #print(np.sum(im == 0))

    return np.sum(im == 0)/totalPixels

#recall, how many positives did we catch
def RecallLabelI(gt,predicted,i):# return the percentage of pixels of class i correctly matched over the total of positives
    sumEquals=np.sum(gt == i)
    if sumEquals!=0:return np.sum(np.logical_and(gt == i, predicted == i)) / sumEquals
    else:return 0

#precision, how many of those predicted are correct
def PrecisionLabelI(gt,predicted,i):# return the percentage of pixels of class i incorrectly matched over the total of positives

    TP=np.sum(np.logical_and(gt == i, predicted == i))
    FP=np.sum(np.logical_and(gt != i, predicted == i))
    if (TP+FP)!=0 : return TP/(TP+FP)
    else: return 0

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    #print("im1: " + repr(im1.shape))
    #print("im2: " + repr(im2.shape))

    if im1.shape != im2.shape:
        print("im1: " + repr(im1.shape))
        print("im1: " + repr(im1.shape))
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

#print ("Dice coefficient between two boolean images")
#print ("-------------------------------------------")



def main(argv):
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    im1 = Image.open(sys.argv[1])
    im2 = Image.open(sys.argv[2])

    # Error control if only one parameter for inverting image is given.
    if len(sys.argv) == 4:
        print("Invert options must be set for both images")
        sys.exit()

    # Inverting input images if is wanted.
    if len(sys.argv) > 3:
        if int(sys.argv[3]) == 1:
            im1 = ImageOps.invert(im1)

        if int(sys.argv[4]) == 1:
            im2 = ImageOps.invert(im2)

    # Image resize for squared images.
    size = 300, 300
    # im1.thumbnail(size, Image.ANTIALIAS)
    # im1.show()
    # im2.thumbnail(size, Image.ANTIALIAS)
    # im2.show()

    # Converting to b/w.
    gray1 = im1.convert('L')
    im1 = gray1.point(lambda x: 0 if x < 128 else 255, '1')
    gray2 = im2.convert('L')
    im2 = gray2.point(lambda x: 0 if x < 128 else 255, '1')

    # Dice coeficinet computation
    res = dice(im1, im2)

    print(res)

if __name__ == "__main__":

    main(sys.argv[1:])
