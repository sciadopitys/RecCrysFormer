#!/bin/bash 

for j in {0..19}
do
    mkdir ../patterson_pt_scaled
    /usr/bin/python3 scaling_tensor_pat.py split_${j}.txt 926.581 -86.3 ${j} &
done