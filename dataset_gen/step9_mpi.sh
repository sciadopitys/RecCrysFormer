#!/bin/bash

for j in {0..19}
do
    mkdir ../electron_density_pt_scaled
    /usr/bin/python3 scaling_tensor_ed.py split_${j}.txt 4.404 -0.324 ${j} &
done
