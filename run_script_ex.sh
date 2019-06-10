#!/bin/bash


# initialise
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ncaplar/.conda/envs/cuda_env/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ncaplar/CodeGpu/software/lib


#dirpath="/tigress/ncaplar/GpuData" 

echo $LD_LIBRARY_PATH

# GPU
gpu_name=0

#general inputs
name="example_results"
repetitions=5

#codes inputs
LClength_in=24
tbin_in=8640000.
#-------------
A_in=30
v_bend_in=2.e-10
a_low_in=1.0
a_high_in=2.0
c_in=0.0
#-------------
PDF_in=1
#-------------
delta1_BPL_in=0.47
delta2_BPL_in=2.53
lambda_s_BPL_in=0.01445
#-------------
lambda_s_LN_in=0.000562341
sigma_LN_in=0.64
#-------------
num_it_in=200
LowerLimit_in=0.00001
UpperLimit_in=10.
LowerLimit_acc_in=0.001
UpperLimit_acc_in=3.
#-------------
len_block_rd_in=1024

for rep in `seq 1 $repetitions`; do

# Create unique results directory for each run
name_file_in=results_${name}_${rep}.bin
echo "results file: $name_file_in"
# Create unique profile file each run
#profFile=prof_${name}_${rep}.nvprof


CUDA_VISIBLE_DEVICES=$gpu_name ./main_cuFFT --LClength $LClength_in --RandomSeed $rep --tbin $tbin_in --A $A_in --v_bend $v_bend_in --a_low $a_low_in --a_high $a_high_in --c $c_in --num_it $num_it_in --LowerLimit $LowerLimit_in --UpperLimit $UpperLimit_in --LowerLimit_acc $LowerLimit_acc_in --UpperLimit_acc $UpperLimit_acc_in --lambda_s_LN $lambda_s_LN_in --sigma_LN $sigma_LN_in --delta1_BPL $delta1_BPL_in --delta2_BPL $delta2_BPL_in --lambda_s_BPL $lambda_s_BPL_in --pdf $PDF_in --len_block_rd $len_block_rd_in --output $name_file_in
done
