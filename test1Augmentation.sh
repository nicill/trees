#!/bin/bash

#CUDA_VISIBLE_DEVICES=2;python tree_detection.py -d /home/owner/Experiments/forests/speciesClassification/trees/Data/Data_Luca_subdivided/ -e 20 -labFus ./speciesConversionTableLuca.txt -aug 6 -dec 30 -th 70
GPU=$1
dataDir=$2
conversion=./speciesConversionTableLuca.txt
softDir=$(pwd)

#try for different augmentations and decreases
for a in 0 2 4 5 8 10 20
do
  for d in 0 10 20 50 70 90
  do
    for th in 0 25 50 70 80 90 95
    do
        echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^starting $a $d $th"
        date
        CUDA_VISIBLE_DEVICES=$GPU;python $softDir/tree_detection.py -d $dataDir -e 20 -labFus $conversion -aug $a -dec $d -th $th
    done
  done
done

#exit()

#code=$2
#th=$3

#make a file for every output (decided perc, accuracy, classwise(prec,rec,dice) )

#echo " $dataDir"

#cd $dataDir

#for f in *;
#do
#    if [ -d "$f" ]; then
	#echo " $f is a directory"
#	python $softDir/evaluateSegmentationResults.py $dataDir/$f/$f"GT.png" $dataDir/$f/$f"ResultTH"$th".png" $code $dataDir/$f/$f"ROI.jpg"
#    fi
#done
