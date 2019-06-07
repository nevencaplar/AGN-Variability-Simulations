# Simulating AGN variability

#### GPU implementation of the light curve simulation algorithm
##### As according to [Sartori et al 2019, submitted to APJ]

The main purpose of this repository it to make available to the public the code explored in Sartori et al 2019. Refer to the main paper for detailed understanding of the implementation and physical choices made.


### Installation:

The code requires MPI to run. In addition, the non-standard libraries that need to be available are: 
 - Random123 (https://github.com/quinoacomputing/Random123)
 - tclap (https://github.com/eile/tclap)
 - lwgrp (https://github.com/LLNL/lwgrp)
 - dtcmp (https://github.com/LLNL/dtcmp)


Follow the instructions to install lwgrp and dtcmp. For example, to install these libraries in my home directory, to the CodeGpu/software directory.

	./configure --prefix='/home/ncaplar/CodeGpu/software/'
	make 
	make install


	./configure --prefix='/home/ncaplar/CodeGpu/software/' --with-lwgrp='/home/ncaplar/CodeGpu/software/'
	make
	make install


