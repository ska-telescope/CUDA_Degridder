
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdlib>
#include <cstdio>
#include <gtest/gtest.h>

#include "degridder.h"
 
// TEST(DegridderTest, VisibilitiesApproximatelyEqual)
// {
// 	double threshold = 1e-5; // 0.00001
// 	double difference = unit_test_generate_approximate_visibilities();
// 	ASSERT_LE(difference, threshold); // diff <= threshold
// }

// int main(int argc, char **argv) {
//     testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }


 int main(int argc, char **argv) {

 	Config config;
	unit_test_init_config(&config);
	int grid_size = config.grid_size;
	Complex *grid = (Complex*) calloc(grid_size * grid_size, sizeof(Complex));
	PRECISION threshold = 1e-5; // 0.00001
	printf("UNIT TEST >>> CONVOLUTION CORRECTION ON GPU....\n\n");
  	
 	if(grid == NULL)
    {	printf("ERROR >>> Unable TO RUN UNIT TEST, failed to allocate Grid...\n\n");
		return EXIT_FAILURE;
    }
 	for(int i = 0; i < grid_size*grid_size; ++i)
 	{
 		grid[i].real = 1.0;
 		grid[i].imag = 0.0;
 	}

    PRECISION rmse = unit_test_gpu_convolution_correction(grid, grid_size, config.cell_size);
	
	if(rmse > threshold)
		printf("UNIT TEST >>> FAILED, RMSE of %f BELOW THRESHOLD %f...\n\n", rmse, threshold);
	else
		printf("UNIT TEST >>> CONVOLUTION CORRECTION UNIT TEST PASSED, RMSE OF %f...\n\n",rmse);


	printf("UNIT TEST >>> Testing DEGRIDDER PIPELINE....\n\n");
	int2 *kernel_supports = (int2*) calloc(config.num_wproj_kernels,sizeof(int2));
    
    if(kernel_supports == NULL)
    {
    	printf("ERROR >>> Unable to allocate required memory for kernel supports, exiting...\n");
		clean_up(NULL, NULL, NULL, NULL, &kernel_supports, NULL);
		return EXIT_FAILURE;
    }

	int total_samples_needed = read_kernel_supports(&config, kernel_supports);
	printf("UPDATE >>> TOTAL KERNEL SAMPLES READ IN %d\n ",total_samples_needed);
	
	if(total_samples_needed <= 0)
	{
		printf("ERROR >>> unable to read samples from the kernel files, exiting...\n");
		clean_up(NULL, NULL, NULL, NULL, &kernel_supports, NULL);
		return EXIT_FAILURE;
	}


	grid = (Complex*) calloc(config.grid_size * config.grid_size, sizeof(Complex));
	Complex *kernel = (Complex*) calloc(total_samples_needed, sizeof(Complex));
	
	// Evaluate memory allocation success
	if(grid == NULL || kernel == NULL)
	{
		printf("ERROR >>> Unable to allocate required memory, exiting...\n");
		clean_up(&grid, NULL, NULL, &kernel, &kernel_supports, NULL);
		return EXIT_FAILURE;
	}
	
	printf("UPDATE >>> Loading kernel...\n");
	// Load in w-projection kernel for w == 0
	bool loaded_kernel = load_kernel(&config, kernel, kernel_supports);
	if(!loaded_kernel)
	{
		clean_up(&grid, NULL, NULL, &kernel, &kernel_supports, NULL);
		return EXIT_FAILURE;
	}
	
	printf(">>> Loading grid...\n");
	// Load data from file
	bool loaded_grid = load_grid(&config, grid);
	printf(">>> Loading visibilities...\n");
	Visibility *vis_uvw = NULL;
	Complex *vis_intensities = NULL;
	bool loaded_vis = load_visibilities(&config, &vis_uvw, &vis_intensities);

	if(!loaded_grid || !loaded_vis || !vis_uvw)
	{	printf("ERROR: Unable to load grid or read visibility files\n\n");
		clean_up(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports, NULL);
		return EXIT_FAILURE;
	}

	printf(">>> Creating seperable 1D prolate and squared l and m...\n");
	// creating 1d seperable prolate spheroidal, stored in first component
	// in second component store one of the l and m values
	PRECISION2 *prolate = (PRECISION2*) calloc(config.grid_size / 2, sizeof(PRECISION2));
	create_1D_half_prolate(prolate, config.grid_size, config.cell_size);
	if(!prolate)
	{
		printf("ERROR: Unable to allocate memory for the 1D prolate spheroidal \n");
	    clean_up(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports, &prolate);
	}

	printf("All data loaded in...\n");


	// Perform degridding to obtain extracted visibility intensities from grid
	execute_degridding(&config, grid, vis_uvw, vis_intensities, 
		config.num_visibilities, prolate, kernel, kernel_supports, total_samples_needed);
	 
    PRECISION maxDiff = unit_test_output_visibilities(&config, vis_uvw, vis_intensities);

    if(maxDiff > threshold)
		printf("UNIT TEST >>> FAILED, MAX DIFFERENCE OF VISIBILITY %f\n BELOW THRESHOLD %f...\n\n", rmse, threshold);
	else
		printf("UNIT TEST >>> DEGRIDDING PIPELINE UNIT TEST PASSED, MAX DIFFERENCE IS %f...\n\n",maxDiff);

    clean_up(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports, &prolate);
    printf(">>> Finished...\n");
    return EXIT_SUCCESS;
 }