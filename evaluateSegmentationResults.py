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
    numClasses=12

    try:
        ROI=(cv2.imread(argv[4],cv2.IMREAD_GRAYSCALE)<100).astype("uint8")
        
    except:
        print("ROI NONE")
        ROI=None

    result=-1
    if code==0:# Evaluate Dice coefficient
        result=dice.dice(mask,gtMask )
        print("*******************************************  dice: "+str(result))
    elif code==1:# Evaluate percentage of matched pixels (includes Background)
        result=dice.equalValue(mask,gtMask,ROI)
        print("*******************************************  % Pixels with equal value (inlcudes BKG class): "+str(result))
    elif code==2:# Evaluate percentage of TPR for each class
        for i in range(numClasses+1):
            result=dice.RecallLabelI(gtMask.copy(),mask.copy(),i,ROI)
            print("******************************************* Class "+str(i)+"  Recall "+str(result))
    elif code==3:# Evaluate precision
        for i in range(numClasses+1):
            result=dice.PrecisionLabelI(gtMask.copy(),mask.copy(),i,ROI)
            print("******************************************* Class "+str(i)+"  Precision: "+str(result))


    return result


"""

    fineMask=cv2.imread(argv[3],0)
    if fineMask is None: raise Exception("No fine mask read")

    for name,mask in [("coarse",coarseMask),("fine",fineMask)]:

        currentDice=dice.dice(mask,255-gtMask )
        currentCoverPerc=dice.coveredPerc(255-gtMask,mask)
        currentFPPerc=dice.FPPerc(255-gtMask,mask)
        print("******************************************* "+name+" mask,  dice: "+str(currentDice)+" and covered Percentage "+str(currentCoverPerc)+" and FP perc "+str(currentFPPerc))
"""



if __name__ == '__main__':
    main(sys.argv)
