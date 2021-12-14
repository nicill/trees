#!/bin/bash

#CUDA_VISIBLE_DEVICES=2;python tree_detection.py -d /home/owner/Experiments/forests/speciesClassification/trees/Data/Data_Luca_subdivided/ -e 20 -labFus ./speciesConversionTableLuca.txt -aug 6 -dec 30 -th 70
GPU=$1
dataDir=$2
whatToDo=$3
prefix=$4

softDir=$(pwd)
conversion=$softDir/speciesConversionTableLuca.txt
outDir=$softDir/output/
codeString="0 1"


for code in $codeString
do
	touch  $outDir"Criterion"$code"pref"$prefix".txt"
	echo "AUGMENT DECREASE THRESHOLD LISTOFSITES " >> $outDir"Criterion"$code"pref"$prefix".txt"
done

# Codify the actions that the script will perform
evaluate=0
compute=0
shutdown=0

if [[ $whatToDo == *"c"* ]]; then
compute=1
fi
if [[ $whatToDo == *"e"* ]]; then
evaluate=1
fi
if [[ $whatToDo == *"s"* ]]; then
shutdown=1
fi



#try for different augmentations and decreases
for a in 0 2 4 5 8 10
do
  for d in 0 10 50 70
  do
    for th in 25 50 90
    do
        echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^starting $a $d $th"
        date
	if [[ $compute = 1 ]];then
	        CUDA_VISIBLE_DEVICES=$GPU;python $softDir/tree_detection.py -d $dataDir -e 20 -labFus $conversion -aug $a -dec $d -th $th -u2 True
	fi
	if [[ $evaluate = 1 ]];then

	
		cd $dataDir
		for code in $codeString
		do
			echo -n	"$a $d $th ">> $outDir"Criterion"$code"pref"$prefix".txt"
			echo -n	"$a $d $th ">> $outDir"Criterion"$code"pref"$prefix"ONE.txt"

			for f in *;
			do
			    if [ -d "$f" ]; then
				echo " $f is a directory"
					outFile=$dataDir/$f/$f"augm"$a"decrease"$d"ResultTH"$th".png"
					outFile2=$dataDir/$f/$f"augm"$a"decrease"$d"ResultTH"$th"ONE.png"
	  			        if [ -f "$outFile" ]; then
						echo " $outFile exists"
						python $softDir/evaluateSegmentationResults.py $dataDir/$f/$f"GT.png" $outFile $code $dataDir/$f/$f"ROI.jpg">> $outDir"Criterion"$code"pref"$prefix".txt"
						python $softDir/evaluateSegmentationResults.py $dataDir/$f/$f"GT.png" $outFile2 $code $dataDir/$f/$f"ROI.jpg">> $outDir"Criterion"$code"pref"$prefix"ONE.txt"
					fi
			    fi
			done
			echo " " >> $outDir"Criterion"$code"pref"$prefix".txt"
			echo " " >> $outDir"Criterion"$code"pref"$prefix"ONE.txt"
		done
	fi	
    done
	
  done
done


if [[ $shutdown = 1 ]];then
shutdown
fi


