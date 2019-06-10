# Simulating AGN variability

#### GPU implementation of the light curve simulation algorithm
##### As according to [Sartori, Trakhtenbrot, Schawinski, Caplar, Treister, Zhang 2019, submitted to APJ]

The main purpose of this repository it to make available to the public the code explored in Sartori et al., 2019. Refer to the main paper for detailed understanding of the implementation and physical choices made.

### Installation:

The code requires MPI to run. In addition, the non-standard libraries that need to be available are: 
 - Random123 (https://github.com/quinoacomputing/Random123)
 - tclap (https://github.com/eile/tclap)
 - lwgrp (https://github.com/LLNL/lwgrp)
 - dtcmp (https://github.com/LLNL/dtcmp)


Follow the instructions to install lwgrp and dtcmp. For example, to install these libraries in my home directory (/home/ncaplar/, in the CodeGpu/software subdirectory) I used:

	./configure --prefix='/home/ncaplar/CodeGpu/software/'
	make 
	make install


	./configure --prefix='/home/ncaplar/CodeGpu/software/' --with-lwgrp='/home/ncaplar/CodeGpu/software/'
	make
	make install

Modify Makefile to point to your /include and /lib directories. As you can see when examining Makefile in this repository, I put Random123 and tclap in the CodeGpu directory (-I/home/ncaplar/CodeGpu/software/Random123-1.09/include -I/home/ncaplar/CodeGpu/tclap/include), while we have just installed stomp and lwgrp in the subdirectory (-L/home/ncaplar/CodeGpu/software/lib). You must also provide path for your cuda and mpi implementation (i.e., the code needs to be able to find -lmpi -lcufft -lcuda -lcudart libraries). 

After modifying Makefile, in the home directory of the reposition, run 

	make
	make install


### Executing:

Find attached a script (run_script_ex.sh) which should simplify running the code. 

First, at the top of the run_script_ex.sh file, I had to add to the LD_LIBRARY_PATH the location of my mpi implementation and the the location for lwgrp/dtcm libraries. This might not be needed in your case, but for me this looks like 

	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ncaplar/.conda/envs/cuda_env/lib
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ncaplar/CodeGpu/software/lib

Now we are ready to create some light-curves! Request some GPUs time on your system, e.g.,

	salloc --gres=gpu:1 -t 00:15:00

I like to use gpustat to verify the name of the machine that I have been given is zero

	gpustat

If you want to run on a machine with a different number change the parameter gpu_name in the run_script_ex script! And now we can execute:

	sh run_script_ex.sh


### Parameters:

Below all of the parameters that are available are described. The values specified below are the ones that are set in the fiducial run_script_ex.sh script. First we start with the names of the generated files

	name="example_results"
	repetitions=5

- name: name of the resulting files
- repetitions: number of light curves created

---

Parameters describing the size of the created light curve

	LClength_in=24
	tbin_in=8640000.


- LClength_in: length of the created light-curve, as the exponent to the power of 2, lengh=2** LClength_in
- tbin_in: time duration of the created light curve

---
Parameters describing the broker power-law PSD of the light curve. The parameters are described with Equation 3 in the paper. 

	A_in=30
	v_bend_in=2.e-10
	a_low_in=1.0
	a_high_in=2.0
	c_in=0.0

- A_in: normalization
- v_bend_in: frequency of the bend of the power-law
- a_low_in: low frequency slope
- a_high_in: high frequency slope
- c_in: offset from zero

---

Parameter which determines if you are using broken power-law or Gaussian description for the PDF. If you are using broken power-law, Gaussian parameters are ignored and vice-versa.

	PDF_in=1

- PDF_in: 1 for broken power-law PDF and 0 for Gaussian PDF

---

Parameters describing the broker power-law PDF for creation of the light curves. The parameters are described with Equation 1 in the paper. 

	delta1_BPL_in=0.47
	delta2_BPL_in=2.53
	lambda_s_BPL_in=0.01445

- delta1_BPL_in: low-Eddington ratio slope
- delta2_BPL_in: high-Eddington ratio slope
- lambda_s_BPL_in: break Eddington ratio where the power law bends

---

Parameters describing the Gaussian (normal) PDF for creation of the light curves The parameters are described with Equation 2 in the paper. 

	lambda_s_LN_in=0.000562341
	sigma_LN_in=0.64

- lambda_s_LN_in: mean of the normal distribution
- sigma_LN_in: width of the normal distribution

---

Parameters describing the limits of the Eddington ratio distribution (PDF). Due to numerical constraints in the random draw algorithm, there are two extra parameters, beyond the lower and upper limit of the distribution, that need to be specified.

	num_it_in=200
	LowerLimit_in=0.00001
	UpperLimit_in=10.
	LowerLimit_acc_in=0.001
	UpperLimit_acc_in=3.

- num_it_in: number of iterative steps, described in Section 3.2.2
- LowerLimit_in: lower limit of the Eddington ratio distribution (PDF) for the random draw algorithm - has to be larger than LowerLimit_acc_in for broken-power law ERDF, and same as UpperLimit_acc_in for Gaussian case!
- UpperLimit_in: upper limit of the Eddington ratio distribution (PDF) for the random draw algorithm - has to be larger than UpperLimit_acc_in for broken-power law ERDF, and same as UpperLimit_acc_in for Gaussian case!
- LowerLimit_acc_in: lower limit of the Eddington ratio distribution (PDF)
- UpperLimit_acc_in: upper limit of the Eddington ratio distribution (PDF)

---

Parameter which determines number of blocks used for the random draw part

	len_block_rd_in=1024

-len_block_rd_in: We recommend not changing this number?

### Examples:

Find 20 run with fiducial parameters (the ones in run_script_ex.sh) in the Examples folder. 
To load the curves use

	import numpy as np
	ER_curve = np.zeros(num_points, dtype = float)
	ER_curve = np.fromfile(<path_to_the_bin_file_here>, dtype = float)

