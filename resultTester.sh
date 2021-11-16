#!/bin/bash

dataDir=$1
code=$2
th=$3

softDir=$(pwd)

echo " $dataDir"

cd $dataDir

for f in *;
do
    if [ -d "$f" ]; then
	#echo " $f is a directory"
	python $softDir/evaluateSegmentationResults.py $dataDir/$f/$f"GT.png" $dataDir/$f/$f"ResultTH"$th".png" $code $dataDir/$f/$f"ROI.jpg"
    fi
done
