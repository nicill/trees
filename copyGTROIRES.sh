#!/bin/bash

dataDir=$1
targetDir=$2

echo " $dataDir"

cd $dataDir

for f in *; 
do
    if [ -d "$f" ]; then
	echo " $f is a directory"
	cp $dataDir/$f/$f"GT.png" $targetDir/$f"GT.png"
	cp $dataDir/$f/$f"Result.png" $targetDir/$f"Result.png"
	cp $dataDir/$f/$f"ROI.jpg" $targetDir/$f"ROI.png"
    fi
done

