#!/bin/bash

for i in {30..40}
do
	python3 /home/alexandre/AutoFX/source/datagen.py --distortion --disto-drive $i --in-path /home/alexandre/dataset/London_Philarmonia_Samples/trumpet/ --out-path "/home/alexandre/dataset/trumpet_distortion/distortion_$i"
done
