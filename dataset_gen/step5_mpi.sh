#!/bin/bash -l

source /path/to/ccp4-<version>/bin/ccp4.setup-sh

for k in {0..19}
do
    export split=${k}
    mkdir 5_electron_density_ccp4_res_${k}
    cat split_${k}.txt | parallel -j 25 'echo -e "LABIN F1=FC PHI=PHIC \n GRID SAMPLE 3.0 \n" | fft hklin 3_electron_density_mtz_res_${split}/{}.gemmi.mtz mapout 5_electron_density_ccp4_res_${split}/{}_fft.ccp4'
done
