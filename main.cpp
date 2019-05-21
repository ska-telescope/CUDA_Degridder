
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
#include <cmath>

#include "degridder.h"

int main(int argc, char **argv)
{
	// Prepare the configuration
	Config config;
	init_config(&config);
	
	// Prepare required memory
	Complex *grid = (Complex*) calloc(config.grid_size * config.grid_size, sizeof(Complex));
	int kernel_size = (int) pow(((config.kernel_size / 2) + 1) * config.oversampling, 2.0);
	Complex *kernel = (Complex*) calloc(kernel_size, sizeof(Complex));
	
	// Evaluate memory allocation success
	if(grid == NULL || kernel == NULL)
	{
		printf("Error: unable to allocate required memory, exiting...\n");
		clean_up(&grid, NULL, NULL, &kernel);
		return EXIT_FAILURE;
	}
	
	printf(">>> Loading kernel...\n");
	// Load in w-projection kernel for w == 0
	bool loaded_kernel = load_kernel(&config, kernel);
	if(!loaded_kernel)
	{
		clean_up(&grid, NULL, NULL, &kernel);
		return EXIT_FAILURE;
	}
	
	printf(">>> Loading grid...\n");
	// Load data from file
	bool loaded_grid = load_grid(&config, grid);
	printf(">>> Loading visibilities...\n");
	Visibility *vis_uvw = NULL;
	Complex *vis_intensities = NULL;
	bool loaded_vis = load_visibilities(&config, &vis_uvw, &vis_intensities);

	if(!loaded_grid || !loaded_vis || vis_uvw == NULL)
	{
		clean_up(&grid, &vis_uvw, &vis_intensities, &kernel);
		return EXIT_FAILURE;
	}
	
	printf("All data loaded in...\n");

	// Perform degridding to obtain extracted visibility intensities from grid
	execute_degridding(&config, grid, vis_uvw, vis_intensities, kernel, config.num_visibilities);
	
	// Save data to file
	save_visibilities(&config, vis_uvw, vis_intensities);
	
	// Free allocated memory
	clean_up(&grid, &vis_uvw, &vis_intensities, &kernel);
	
	printf(">>> Finished...\n");

	return EXIT_SUCCESS;
}

