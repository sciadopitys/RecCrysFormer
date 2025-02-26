#!/bin/bash 

for j in {0..19}
do
    mkdir 6_patterson_pt_${j}
    /usr/bin/python3 convert_ccp4_to_pt_pat.py split_${j}.txt ${j} &
done