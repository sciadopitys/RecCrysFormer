#!/bin/bash -l

source /path/to/ccp4-<version>/bin/ccp4.setup-sh

for j in {0..19}
do
    export split=${j}
    mkdir 4_patterson_ccp4_res_${j}
    cat split_${j}.txt | parallel -j 25 'echo -e "LABIN F1=FC PHI=PHIC \n GRID SAMPLE 3.0 \n PATT \n" | fft hklin 3_electron_density_mtz_res_${split}/{}.gemmi.mtz mapout 4_patterson_ccp4_res_${split}/{}_patterson.ccp4'
done
    