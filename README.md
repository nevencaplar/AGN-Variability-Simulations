# Simulating AGN variability

#### GPU implementation of the light curve simulation algorithm
##### As according to [Sartori, Trakhtenbrot, [Schawinski](https://github.com/kevinschawinski), [Caplar](https://github.com/nevencaplar), Treister, [Zhang](https://github.com/DS3Lab) 2019, submitted to APJ]

The main purpose of this repository it to make available to the public the code explored in Sartori et al., 2019. Refer to the main paper for detailed understanding of the implementation and physical choices made. This is GPU implementation of the code described in [Emmanoulopoulos et al., 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.433..907E/abstract). Implementation in pure Python is available [here](https://github.com/samconnolly/DELightcurveSimulation).

### Installation:

The code requires CUDA to run. In addition, the non-standard libraries (some of which require MPI) that need to be available are: 
 - Random123 (https://github.com/quinoacomputing/Random123)
 - tclap (https://github.com/eile/tclap)
 - lwgrp (https://github.com/LLNL/lwgrp)
 - dtcmp (https://github.com/LLNL/dtcmp)


After installing Random123 and tclap, follow the instructions on respective GitHub pages to install lwgrp and dtcmp. These are the packages that demand MPI and dtcmp depends on lwgrp. For example, to install these libraries in my home directory (/home/ncaplar/, in the CodeGpu/software subdirectory) I used:

	./configure --prefix='/home/ncaplar/CodeGpu/software/'
	make 
	make install


	./configure --prefix='/home/ncaplar/CodeGpu/software/' --with-lwgrp='/home/ncaplar/CodeGpu/software/'
	make
	make install

Modify Makefile to point to your /include and /lib directories. As you can see when examining Makefile in this repository, I put Random123 and tclap in the CodeGpu directory (-I/home/ncaplar/CodeGpu/software/Random123-1.09/include -I/home/ncaplar/CodeGpu/tclap/include), while we have just installed dtcmp and lwgrp in the software subdirectory (-L/home/ncaplar/CodeGpu/software/lib). You must also provide path for your cuda and mpi implementation (i.e., the code needs to be able to find -lmpi -lcufft -lcuda -lcudart libraries). 

After modifying the provided Makefile (in the home directory of the repository) with the paths to your installations, run 

	make

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
- tbin_in: time duration of a single time step, i.e., the resolution of the light-curve. Nominally it is given in seconds. 

---
Parameters describing the broker power-law PSD of the light curve. The parameters are described with Equation 3 in the paper. 

	A_in=30
	v_bend_in=2.e-10
	a_low_in=1.0
	a_high_in=2.0
	c_in=0.0

- A_in: dummy variable for normalization - note that the code creates light curves whose PSD shape is consistent with the input shape, but the normalization effectively depends on the PDF (see Equation A5)
- v_bend_in: frequency of the bend of the power-law
- a_low_in: low frequency slope
- a_high_in: high frequency slope
- c_in: offset from zero

---

Parameter which determines if you are using broken power-law or log-normal description for the PDF. If you are using broken power-law, log-normal parameters are ignored and vice-versa.

	PDF_in=1

- PDF_in: 1 for broken power-law PDF and 0 for log-normal PDF

---

Parameters describing the broker power-law PDF for creation of the light curves. The parameters are described with Equation 1 in the paper. 

	delta1_BPL_in=0.47
	delta2_BPL_in=2.53
	lambda_s_BPL_in=0.01445

- delta1_BPL_in: low-Eddington ratio slope
- delta2_BPL_in: high-Eddington ratio slope
- lambda_s_BPL_in: break Eddington ratio where the power law bends

---

Parameters describing the log-normal PDF for creation of the light curves The parameters are described with Equation 2 in the paper. 

	lambda_s_LN_in=0.000562341
	sigma_LN_in=0.64

- lambda_s_LN_in: mean of the log-normal distribution. Note that this is linear value for the mean of the distribution, i.e., lamda^star=lambda_s_LN_in and NOT log(lamda^star)= lambda_s_LN_in.
- sigma_LN_in: width of the log-normal distribution

---

Parameters describing the limits of the Eddington ratio distribution (PDF). Due to numerical constraints in the random draw algorithm, there are two extra parameters, beyond the lower and upper limit of the distribution, that need to be specified.

	num_it_in=200
	LowerLimit_in=0.00001
	UpperLimit_in=10.
	LowerLimit_acc_in=0.001
	UpperLimit_acc_in=3.

- num_it_in: number of iterative steps, described in Section 3.2.2
- LowerLimit_in: lower limit of the Eddington ratio distribution (PDF) for the random draw algorithm - has to be smaller than LowerLimit_acc_in for broken-power law ERDF, and same as UpperLimit_acc_in for log-normal case!
- UpperLimit_in: upper limit of the Eddington ratio distribution (PDF) for the random draw algorithm - has to be larger than UpperLimit_acc_in for broken-power law ERDF, and same as UpperLimit_acc_in for log-normal case!
- LowerLimit_acc_in: lower limit of the Eddington ratio distribution (PDF)
- UpperLimit_acc_in: upper limit of the Eddington ratio distribution (PDF)

---

Parameter which determines number of blocks used for the random draw part

	len_block_rd_in=1024

- len_block_rd_in: In general, we recommend not changing this number unless you know what you are doing. However, if you change this number it has to number that is a power of 2 (512, 1024, 2048...) and such that such that LC lenght /len_block_rd is smaller than the number of blocks in the GPU that you are using. Changing the number will change the performance of the code. In general, for the random draw step, we have to create 1 random seed per block (“parallel element”). This operation is quite expensive, both time-wise and memory-wise, so we want to reduce the number of blocks. On the other hand, having fewer blocks means that we have fewer parallel operations.
So, one has to find the right balance between these two constrains, depending on the other properties of the generated light-curves. 

In the `run_script_ex.sh you can uncomment the following line, which will create a profile file that is the “standard CUDA file” used for debugging and where you can look up how much time every step takes.

	#Create unique profile file each run
	#profFile=prof_${name}_${rep}.nvprof


### Loading the data and examples:

Find 20 run with fiducial parameters (the ones in run_script_ex.sh) in the Examples folder. 
To load the curves use

	import numpy as np
	ER_curve = np.zeros(num_points, dtype = float)
	ER_curve = np.fromfile(<path_to_the_bin_file_here>, dtype = float)

