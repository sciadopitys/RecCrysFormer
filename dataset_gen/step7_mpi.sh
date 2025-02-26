#!/bin/bash 

for j in {0..19}
do
    mkdir 7_electron_density_pt_${j}
    /usr/bin/python3 convert_ccp4_to_pt_ed.py split_${j}.txt ${j} &
done
