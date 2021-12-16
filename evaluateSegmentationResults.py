import numpy as np
import sys
import dice
import cv2


def main(argv):

    gtMask=cv2.imread(argv[1],cv2.IMREAD_GRAYSCALE).astype("uint8")
    if gtMask is None: raise Exception("No ground Truth mask read")

    #print(np.unique(gtMask))

    mask=cv2.imread(argv[2],cv2.IMREAD_GRAYSCALE).astype("uint8")
    if mask is None: raise Exception("No coarse mask read")

    #print(np.unique(mask))

    code=int(argv[3])
    numClasses=6 #this is the number of the bigger
    #class present (classes 0-numClasses both included)

    try:
        if code==0: ROI=(cv2.imread(argv[4],cv2.IMREAD_GRAYSCALE)<100).astype("uint8")
        else:ROI=(cv2.imread(argv[4],cv2.IMREAD_GRAYSCALE)<100)
    except Exception as e:
        print("ROI NONE "+argv[4]+" "+str(e))
        ROI=None

    #erase anything outside the ROI!!!!
    gtMask[ROI==0]=0
    mask[ROI==0]=0

    result=-1
    if code==0: # Percentage of decided Pixels
        result=dice.decidedPercentage(mask.copy(),gtMask.copy(),ROI)
    elif code==1: # Evaluate percentage of matched pixels (includes Background but not undecided pixels)
        result=dice.equalValue(mask.copy(),gtMask.copy(),ROI)
        #print("*******************************************  % Pixels with equal value (inlcudes BKG class): "+str(result))
    elif code==2:# Evaluate percentage of TPR for each class
        result=[]
        for i in range(numClasses+1):
            aux1=gtMask.copy()
            aux2=mask.copy()
            result.append(dice.RecallLabelI(aux1[ROI],aux2[ROI],i))
            #print("******************************************* Class "+str(i)+"  Recall "+str(result))
    elif code==3:# Evaluate precision for each class
        result=[]
        for i in range(numClasses+1):
            result.append(dice.PrecisionLabelI(gtMask.copy()[ROI],mask.copy()[ROI],i))
            #print("******************************************* Class "+str(i)+"  Precision: "+str(result))
    elif code==4:# Evaluate Dice for each class
        result=[]
        for i in range(numClasses+1):
            result.append(dice.DiceLabelI(gtMask.copy()[ROI],mask.copy()[ROI],i))
            #print("******************************************* Class "+str(i)+"  Dice: "+str(result))
    elif code==5:# Evaluate correctly class in each class
        mask[ROI==0]=255

        result=[]
        for i in range(numClasses+1):
            result.append(dice.CorrectLabelI(gtMask.copy()[ROI],mask.copy()[ROI],i))
    elif code==6:# Evaluate correctly class in each class

        result=[]
        for i in range(numClasses+1):
            result.append(dice.totalLabelI(gtMask.copy()[ROI],mask.copy()[ROI],i))

    if isinstance(result,list):
        for x in result:print(str(x)+str(" "),end="")
        #print("")
    else:print(str(result)+" ",end="")

    return result


if __name__ == '__main__':
    main(sys.argv)
