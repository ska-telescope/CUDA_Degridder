
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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <numeric>

#include "degridder.h"

void init_config(Config *config)
{
	// Single dimension of grid (dirty residual image)
	config->grid_size = 18000;
	
	config->right_ascension = true;

	config->force_zero_w_term = false;
	
	config->cell_size = 6.39708380288950e-6;
	
	config->frequency_hz = 100e6;
	
	// Single dimension of basic convolution kernel
	config->kernel_size = 9;

	// Specify the number of kernels used in w projection
	config->num_wproj_kernels = 339;

	config->max_w = 7083.386050;

	config->w_scale = pow(config->num_wproj_kernels - 1, 2.0) / config->max_w;
	
	// Kernel oversampling factor
	config->oversampling = 4; // Oxford configuration
	
	// Used to convert visibility uvw coordinates into grid coordinates
	config->uv_scale = config->grid_size * config->cell_size;
	
	// Number of visibilities to process
	config->num_visibilities = 100;
	
	// File location to load grid
	config->grid_real_source_file = "../unit_test_data/ModelImage_18000.csv";

	// File location to load pre-calculated w-projection kernel
	config->kernel_real_source_file = "../unit_test_data/kernels/el82-el70_kernel_real.csv";
	config->kernel_imag_source_file = "../unit_test_data/kernels/el82-el70_kernel_imag.csv";

	// Specify file which holds the supports for all kernels
	config->kernel_support_file = "../unit_test_data/kernels/el82-el70_kernel_supports.csv";

	// File location to load visibility uvw coordinates  
	config->visibility_source_file = "../unit_test_data/el82-70_unit_test.csv";   
	
	// File location to store extracted visibilities  
	config->visibility_dest_file = "../el82-70_output_degridder.csv";

	// Number of CUDA threads per block - this is GPU specific
	config->gpu_max_threads_per_block = 1024;

	//Number of CUDA threads per block dimension in x and y - this is GPU specific and used for FFT and CC
	config->gpu_max_threads_per_block_dimension = 32;

	// Enable/disable CUDA timing of degridding kernel 
	config->time_degridding = true;

	// Perform convolution correction and FFT pre-processing on input grid
	config->conv_correction_and_fft = true;

	config->conv_correction_performed = false;
}
void execute_degridding(Config *config, Complex *grid, 
	Visibility *vis_uvw, Complex *vis_intensities, int num_visibilities, double2* prolate, 
	Complex *kernel, int2 *kernel_supports, int num_kernel_samples)
{
	bool perform_cc_and_fft = config->conv_correction_and_fft;
	bool cc_performed = config->conv_correction_performed;
	cudaEvent_t start, stop;
	// Handles for GPU memory
	double2 *d_grid;
	double2 *d_kernel;
	double3 *d_vis_uvw;
	double2 *d_vis;
	double2 *d_prolate;
	int2 *d_supports;


	printf("Binding grid to GPU...\n");	
	// Allocate and copy grid to GPU
	int grid_size_square = config->grid_size * config->grid_size;
	CUDA_CHECK_RETURN(cudaMalloc(&d_grid, sizeof(double2) * grid_size_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_grid, grid, sizeof(double2) * grid_size_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();


	// Perform convolution correction on host based grid
	if(perform_cc_and_fft && !cc_performed)
	{
		CUDA_CHECK_RETURN(cudaMalloc(&d_prolate, sizeof(double2) * config->grid_size/2));
		CUDA_CHECK_RETURN(cudaMemcpy(d_prolate, prolate, sizeof(double2) * config->grid_size/2, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->grid_size);
		int num_blocks_per_dimension = (int) ceil((double) config->grid_size / max_threads_per_block_dimension);
		dim3 cc_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
		dim3 cc_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
		// Perform 2D FFT shift

		if(config->time_degridding)
		{	
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
		}

		execute_convolution_correction<<<cc_blocks, cc_threads>>>(d_grid, d_prolate, config->grid_size);

		if(config->time_degridding)
		{
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf(">>> GPU accelerated Convolution Correction completed in %f milliseconds...\n\n", milliseconds);
		}

		cudaDeviceSynchronize();
		printf(">>> CONVOLUTION CORRECTION DONE \n...");
		config->conv_correction_performed = true;
		CUDA_CHECK_RETURN(cudaFree(d_prolate));
	}


	// Perform FFT on device based grid
	if(perform_cc_and_fft)
	{
		printf(">>> PERFORMING FFT\n...");
		if(config->time_degridding)
		{	
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
		}
		execute_CUDA_FFT(config, d_grid);
		if(config->time_degridding)
		{
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf(">>> GPU accelerated FFT completed in %f milliseconds...\n\n", milliseconds);
		}
	}

	printf("Grid bound to GPU...\n");
	// Allocate and copy kernel to device
	//int oversampled_half_kernel = config->oversampling * ((config->kernel_size / 2) + 1);
	//int total_kernel_size = oversampled_half_kernel * oversampled_half_kernel;

	CUDA_CHECK_RETURN(cudaMalloc(&d_kernel, sizeof(double2) * num_kernel_samples));
	CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, kernel, sizeof(double2) * num_kernel_samples,
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	//bind the supports array

	CUDA_CHECK_RETURN(cudaMalloc(&d_supports, sizeof(int2) * config->num_wproj_kernels));
	CUDA_CHECK_RETURN(cudaMemcpy(d_supports, kernel_supports, sizeof(int2) * config->num_wproj_kernels,
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	printf("Kernels bound to GPU...\n");
	// Allocate and copy visibility uvw to device
	CUDA_CHECK_RETURN(cudaMalloc(&d_vis_uvw, sizeof(double3) * num_visibilities));
	CUDA_CHECK_RETURN(cudaMemcpy(d_vis_uvw, vis_uvw, sizeof(double3) * num_visibilities,
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	printf("Vis UVW bound to GPU...\n");

	// Allocate memory on device for storing extracted complex visibilities
	CUDA_CHECK_RETURN(cudaMalloc(&d_vis, sizeof(double2) * num_visibilities));
	cudaDeviceSynchronize();
	printf("Vis memory allocated on GPU...\n");

	int max_threads_per_block = min(config->gpu_max_threads_per_block, num_visibilities);
	int num_blocks = (int) ceil((double) num_visibilities / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	printf(">>> Going to use %d number of blocks, %d threads, for %d visibilities...\n",
		num_blocks, max_threads_per_block, num_visibilities);

	// Optional timing functionality
	if(config->time_degridding)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}



	printf("Beginning kernel...\n");
	// Execute degridding kernel
	degridding<<<kernel_blocks, kernel_threads>>>(d_grid, d_kernel, d_supports, 
		d_vis_uvw, d_vis, num_visibilities, config->oversampling,
		config->grid_size, config->uv_scale, config->w_scale);
	cudaDeviceSynchronize();
	printf("Finished kernel...\n");

	// Optional report on timing
	if(config->time_degridding)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf(">>> GPU accelerated degridding completed in %f milliseconds...\n\n", milliseconds);
	}

	// Copy back predicted complex visibilities to host
	CUDA_CHECK_RETURN(cudaMemcpy(vis_intensities, d_vis,
		num_visibilities * sizeof(double2), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// Clean up
	CUDA_CHECK_RETURN(cudaFree(d_grid));
	CUDA_CHECK_RETURN(cudaFree(d_kernel));
	CUDA_CHECK_RETURN(cudaFree(d_vis_uvw));
	CUDA_CHECK_RETURN(cudaFree(d_vis));
	CUDA_CHECK_RETURN(cudaFree(d_supports));
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

int read_kernel_supports(Config *config, int2 *kernel_supports)
{
	int total_kernel_samples_needed = 0;

	FILE *kernel_support_file = fopen(config->kernel_support_file,"r");

	if(kernel_support_file == NULL)
	{
		return -1;
	}


	for(int plane_num = 0; plane_num < config->num_wproj_kernels; ++plane_num)
	{
		fscanf(kernel_support_file,"%d\n",&(kernel_supports[plane_num].x));
		kernel_supports[plane_num].y = total_kernel_samples_needed;
		total_kernel_samples_needed += (int)pow((kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);

		//printf("%d) %d support with offset at %d \n",plane_num,kernel_supports[plane_num].x,kernel_supports[plane_num].y);
	}

	fclose(kernel_support_file);
	return total_kernel_samples_needed;
}
 

__global__ void fftshift_2D(double2 *grid, const int width)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= width || col_index >= width)
        return;
 
    double a = 1 - 2 * ((row_index + col_index) & 1);
    grid[row_index * width + col_index].x *= a;
    grid[row_index * width + col_index].y *= a;
}

__device__ double2 complex_mult(const double2 z1, const double2 z2)
{
	return make_double2(z1.x * z2.x - z1.y * z2.y, z1.y * z2.x + z1.x * z2.y);
}


__global__ void degridding(const double2 *grid, const double2 *kernel, const int2 *supports,
	const double3 *vis_uvw, double2 *vis, const int num_vis, const int oversampling,
	const int grid_size, const double uv_scale, const double w_scale)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= num_vis)
		return;

	// Represents index of w-projection kernel in supports array
	const int plane_index = (int) round(sqrt(fabs(vis_uvw[vis_index].z * w_scale)));

	// Scale visibility uvw into grid coordinate space
	const double2 grid_coord = make_double2(
		vis_uvw[vis_index].x * uv_scale,
		vis_uvw[vis_index].y * uv_scale
	);

	const int half_grid_size = grid_size / 2;
	const int half_support = supports[plane_index].x;

	double conjugate = (vis_uvw[vis_index].z < 0.0) ? -1.0 : 1.0;
	conjugate *= -1.0;
	//HAD TO REVERSE THE CONJUGATE FOR DEGRIDDING..
	//double conjugate = (vis_uvw[vis_index].z > 0.0) ? -1.0 : 1.0;


	const double2 snapped_grid_coord = make_double2(
		round(grid_coord.x * oversampling) / oversampling,
		round(grid_coord.y * oversampling) / oversampling
	);

	const double2 min_grid_point = make_double2(
		ceil(snapped_grid_coord.x - half_support),
		ceil(snapped_grid_coord.y - half_support)
	);

	const double2 max_grid_point = make_double2(
		floor(snapped_grid_coord.x + half_support),
		floor(snapped_grid_coord.y + half_support)
	);

	double2 grid_point = make_double2(0.0, 0.0);
	double2 deconvolved = make_double2(0.0, 0.0);
	double2 predicted_vis = make_double2(0.0, 0.0);
	double2 kernel_sample = make_double2(0.0, 0.0);
	int2 kernel_uv_index = make_int2(0, 0);

	int grid_index = 0;
	int kernel_index = 0;
	int w_kernel_offset = supports[plane_index].y;

	// Iteratively extract predicted visibility from grid
	for(int grid_v = min_grid_point.y; grid_v <= max_grid_point.y; ++grid_v)
	{	
		kernel_uv_index.y = abs((int)round((grid_v - snapped_grid_coord.y) * oversampling));
		
		for(int grid_u = min_grid_point.x; grid_u <= max_grid_point.x; ++grid_u)
		{
			kernel_uv_index.x = abs((int)round((grid_u - snapped_grid_coord.x) * oversampling));

			kernel_index = w_kernel_offset + kernel_uv_index.y * (half_support + 1)
				* oversampling + kernel_uv_index.x;
			kernel_sample = make_double2(kernel[kernel_index].x, kernel[kernel_index].y  * conjugate);

			grid_index = (grid_v + half_grid_size) * grid_size + (grid_u + half_grid_size);

			double2 gridPoint = make_double2(grid[grid_index].x, grid[grid_index].y);

			deconvolved = complex_mult(gridPoint, kernel_sample);

			predicted_vis.x += deconvolved.x;
			predicted_vis.y += deconvolved.y;
		}
	}


	vis[vis_index] = make_double2(predicted_vis.x, predicted_vis.y);
}
 

void save_visibilities(Config *config, Visibility *vis_uvw, Complex *vis_intensity)
{
	FILE *vis_file = fopen(config->visibility_dest_file, "w");
	
	if(vis_file == NULL)
	{
		printf("Unable to open file...\n");
		return; // unsuccessfully saved visibility data
	}
	
	// Define the number of processed visibilities
	fprintf(vis_file, "%d\n", config->num_visibilities);
	
	double meters_to_wavelengths = config->frequency_hz / C;
	Visibility current_vis;
	Complex current_intensity;
	
	for(int vis_index = 0; vis_index < config->num_visibilities; ++vis_index)
	{
		current_vis = vis_uvw[vis_index];
		current_intensity = vis_intensity[vis_index];
		
		current_vis.u /= meters_to_wavelengths;
		current_vis.v /= meters_to_wavelengths;
		current_vis.w /= meters_to_wavelengths;
		
		if(config->right_ascension)
		{
			current_vis.u *= -1.0;
			current_vis.w *= -1.0;
		}
		
		// u, v, w, vis(real), vis(imag), weighting
		fprintf(vis_file, "%.15f %.15f %.15f %.15f %.15f %.15f\n", 
			current_vis.u,
			current_vis.v,
			current_vis.w,
			current_intensity.real,
			current_intensity.imag,
			1.0); // static weight (for now)
	}
	
	fclose(vis_file);
}

bool load_kernel(Config *config, Complex *kernel, int2 *kernel_supports)
{
	FILE *kernel_real_file = fopen(config->kernel_real_source_file, "r");
	FILE *kernel_imag_file = fopen(config->kernel_imag_source_file, "r");
	
	if(kernel_real_file == NULL || kernel_imag_file == NULL)
	{
		printf("ERROR >>> Unable to open kernel source files...\n");
		if(kernel_real_file != NULL) fclose(kernel_real_file);
		if(kernel_imag_file != NULL) fclose(kernel_imag_file);
		return false; // unsuccessfully loaded data
	}
	
	int kernel_index = 0;

	for(int plane_num = 0; plane_num < config->num_wproj_kernels; ++plane_num)
	{
		int number_samples_in_kernel = (int) pow((kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);

		for(int sample_number = 0; sample_number < number_samples_in_kernel; ++sample_number)
		{	
			float real = 0.0;
			float imag = 0.0;

			fscanf(kernel_real_file, "%f ", &real);
			fscanf(kernel_imag_file, "%f ", &imag);

			kernel[kernel_index] = (Complex) {.real = real, .imag = imag};
			kernel_index++;
		}

	}
	printf("Total number of samples read %d \n",kernel_index);

	fclose(kernel_real_file);
	fclose(kernel_imag_file);
	return true;
}


bool load_grid(Config *config, Complex *grid)
{
	FILE *grid_real_file = fopen(config->grid_real_source_file, "r");
	
	if(grid_real_file == NULL)
	{
		printf("Unable to open grid files...\n");
		if(grid_real_file != NULL) fclose(grid_real_file);
		return false; // unsuccessfully loaded data
	} 
	
	int grid_size = config->grid_size;
	int grid_index = 0;
	double grid_real = 0.0;
	
	double valuesum = 0.0;

	for(int row_index = 0; row_index < grid_size; ++row_index)
	{
		for(int col_index = 0; col_index < grid_size; ++col_index)
		{
			if(col_index < grid_size-1)
			{
				fscanf(grid_real_file, "%lf,", &grid_real);
			}
			else
			{
				fscanf(grid_real_file, "%lf\n", &grid_real);
			}

			valuesum += grid_real;
			grid_index = row_index * grid_size + col_index;
			grid[grid_index] = (Complex) {.real = grid_real, .imag = 0.0};
		}
	}
	printf("UPDATE >>> Successfully read in %f source intensity sum \n",valuesum);
	fclose(grid_real_file);
	return true; 
}

bool load_visibilities(Config *config, Visibility **vis_uvw, Complex **vis_intensities)
{
	// Attempt to open visibility source file
	FILE *vis_file = fopen(config->visibility_source_file, "r");
	if(vis_file == NULL)
	{
		printf("Unable to open visibility file...\n");
		return false; // unsuccessfully loaded data
	}
	
	// Configure number of visibilities from file
	int num_vis = 0;
	fscanf(vis_file, "%d", &num_vis);
	config->num_visibilities = num_vis;

	// Allocate memory for incoming visibilities
	*vis_uvw = (Visibility*) calloc(num_vis, sizeof(Visibility));
	*vis_intensities = (Complex*) calloc(num_vis, sizeof(Complex));
	if(*vis_uvw == NULL || *vis_intensities == NULL)
	{
		printf("Unable to allocate memory...\n");
		fclose(vis_file);
		return false;
	}
	
	// Load visibility uvw coordinates into memory
	double vis_u = 0.0;
	double vis_v = 0.0;
	double vis_w = 0.0;
	double vis_real = 0.0;
	double vis_imag = 0.0;
	double vis_weight = 0.0;
	double meters_to_wavelengths = config->frequency_hz / C;


	for(int vis_index = 0; vis_index < num_vis; ++vis_index)
	{
		// Discard vis(real), vis(imag), and weighting (for now)
		fscanf(vis_file, "%lf %lf %lf %lf %lf %lf\n", &vis_u, &vis_v,
			&vis_w, &vis_real, &vis_imag, &vis_weight);
	
		(*vis_uvw)[vis_index] = (Visibility) {
			.u = vis_u * meters_to_wavelengths,
			.v = vis_v * meters_to_wavelengths,
			.w = (config->force_zero_w_term) ? 0.0 : vis_w * meters_to_wavelengths 
		};

		if(config->right_ascension)
		{
			(*vis_uvw)[vis_index].u *= -1.0;
			(*vis_uvw)[vis_index].w *= -1.0;
		}
	}
	// Clean up
	fclose(vis_file);
	return true;
}

// Calculates a sample on across a prolate spheroidal
// Note: this is the Fred Schwabb approximation technique
double calc_spheroidal_sample(double nu)
{
    static double p[] = {0.08203343, -0.3644705, 0.627866, -0.5335581, 0.2312756,
        0.004028559, -0.03697768, 0.1021332, -0.1201436, 0.06412774};
    static double q[] = {1.0, 0.8212018, 0.2078043,
        1.0, 0.9599102, 0.2918724};

    int part = 0;
    int sp = 0;
    int sq = 0;
    double nuend = 0.0;
    double delta = 0.0;
    double top = 0.0;
    double bottom = 0.0;

    if(nu >= 0.0 && nu < 0.75)
    {
        part = 0;
        nuend = 0.75;
    }
    else if(nu >= 0.75 && nu < 1.0)
    {
        part = 1;
        nuend = 1.0;
    }
    else
        return 0.0;

    delta = nu * nu - nuend * nuend;
    sp = part * 5;
    sq = part * 3;
    top = p[sp];
    bottom = q[sq];

    for(int i = 1; i < 5; i++)
        top += p[sp+i] * pow(delta, i);
    for(int i = 1; i < 3; i++)
        bottom += q[sq+i] * pow(delta, i);
    return (bottom == 0.0) ? 0.0 : top/bottom;
}

void create_1D_half_prolate(double2 *prolate, int grid_size, double cell_size)
{
	int grid_half_size = grid_size / 2;
	double nu = 0.0;
	double taper = 0.0;
	double l_square = 0.0;

	for(int index = 0; index < grid_half_size; ++index)
	{
		nu = ((double)index / (double)grid_half_size);
		taper = calc_spheroidal_sample(nu);
		prolate[index].x = taper;
		l_square = pow((double)index * cell_size, 2.0);
		prolate[index].y = l_square;
	}
}


__global__ void execute_convolution_correction(double2 *grid, const double2 *prolate, const int grid_size)
{
	const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(row_index >= grid_size || col_index >= grid_size)
    	return;

    const int grid_index = row_index * grid_size + col_index;
    const int half_grid_size = grid_size / 2;

    const double taper = prolate[abs(col_index - half_grid_size)].x * prolate[abs(row_index - half_grid_size)].x;

    const double l = prolate[abs(col_index - half_grid_size)].y;
    const double m = prolate[abs(row_index - half_grid_size)].y;

    grid[grid_index].x = (fabs(taper) > (1E-10)) ? grid[grid_index].x / (taper * sqrt(1.0 - l - m)) : 0.0;
}

void execute_CUDA_FFT(Config *config, double2 *grid)
{
	int grid_size = config->grid_size;

	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, grid_size);
	int num_blocks_per_dimension = (int) ceil((double) grid_size / max_threads_per_block_dimension);
	dim3 shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("Shifting grid data for 2D FFT...\n");
	// Perform 2D FFT shift
	fftshift_2D<<<shift_blocks, shift_threads>>>(grid, grid_size);
	cudaDeviceSynchronize();

	printf("Performing 2D FFT...\n");
	// Perform 2D FFT
	cufftHandle fft_plan;
	CUFFT_SAFE_CALL(cufftPlan2d(&fft_plan, grid_size, grid_size, CUFFT_Z2Z));
	CUFFT_SAFE_CALL(cufftExecZ2Z(fft_plan, grid, grid, CUFFT_FORWARD));
	cudaDeviceSynchronize();

	printf("Shifting grid data back into place...\n");
	// Perform 2D FFT shift back
	fftshift_2D<<<shift_blocks, shift_threads>>>(grid, grid_size);
	cudaDeviceSynchronize();
}

void clean_up(Complex **grid, Visibility **vis_uvw, Complex **vis_intensities, Complex **kernel, int2 **kernel_supports, double2 **prolate)
{
	printf(">>> Cleaning up allocated host memory...\n");
	if(*grid) 			 free(*grid);
	if(*vis_uvw) 	 	 free(*vis_uvw);
	if(*vis_intensities) free(*vis_intensities);
	if(*kernel) 		 free(*kernel);
	if(*kernel_supports) free(*kernel_supports);
	if(*prolate) 		 free(*prolate);
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

static void cufft_safe_call(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		printf("CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n",
			__FILE__, __LINE__, err, cuda_get_error_enum(err));
		cudaDeviceReset();
    }
}

static const char* cuda_get_error_enum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}


/***************************************
*      UNIT TESTING FUNCTIONALITY      *
***************************************/

//Unit test for l
double unit_test_gpu_convolution_correction(Complex *grid, int grid_size, double cell_size)
{
	double2 *d_grid;
	double2 *d_prolate;

	double2 *prolate = (double2*) calloc(grid_size / 2,sizeof(double2));
	create_1D_half_prolate(prolate, grid_size, cell_size);

	int grid_size_square = grid_size * grid_size;
	Complex *outputGrid = (Complex*) calloc(grid_size_square, sizeof(Complex));


	printf("UNIT TEST >>> COPYING PROLATE TO GPU BUFFER..\n");
	CUDA_CHECK_RETURN(cudaMalloc(&d_prolate, sizeof(double2) * grid_size/2));
	CUDA_CHECK_RETURN(cudaMemcpy(d_prolate, prolate, sizeof(double2) * grid_size/2, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	printf("UNIT TEST >>> COPYING GRID TO GPU...\n");
	
	CUDA_CHECK_RETURN(cudaMalloc(&d_grid, sizeof(Complex) * grid_size_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_grid, grid, sizeof(Complex) * grid_size_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	printf("UNIT TEST >>> EXECUTING CPU CONVOLUTION CORRECTION...\n");
	execute_convolution_correction_cpu(grid, grid_size, cell_size);

	printf("UNIT TEST >>> EXECUTING GPU CONVOLUTION CORRECTION..\n");

	int max_threads_per_block_dimension = min(32, grid_size);
	int num_blocks_per_dimension = (int) ceil((double) grid_size / max_threads_per_block_dimension);
	dim3 cc_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 cc_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
	execute_convolution_correction<<<cc_blocks, cc_threads>>>(d_grid, d_prolate, grid_size);
	cudaDeviceSynchronize();
	printf("UNIT TEST >>> CONVOLUTION CORRECTION DONE, COPYING GRID BACK ...\n");

	CUDA_CHECK_RETURN(cudaMemcpy(outputGrid, d_grid,
		grid_size_square * sizeof(Complex), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	printf("UNIT TEST >>> CALCULATE RMSE OF GPU vs GPU ...\n");

	double sumSquareDiff;
	double sumSquareCPU;
	for(int i = 0; i < grid_size_square; i++)
	{	sumSquareDiff += pow(abs(grid[i].real-outputGrid[i].real)+abs(grid[i].imag-outputGrid[i].imag),2.0);
		sumSquareCPU += pow(abs(grid[i].real)+abs(grid[i].imag),2.0);
	}

	double rmse = sqrt(sumSquareDiff) / sqrt(sumSquareCPU);

	grid = &(*outputGrid);

	printf("UNIT TEST >>> CLEAN UP USED MEMORY ...\n");
	CUDA_CHECK_RETURN(cudaFree(d_prolate));
	CUDA_CHECK_RETURN(cudaFree(d_grid));
	free(outputGrid);

	return rmse;
}


// TODO: This could be parallelized...
void execute_convolution_correction_cpu(Complex *grid, double grid_size, double cell_size)
{
	double grid_half_size = grid_size / 2;
	double nu_y = 0.0;
	double nu_x = 0.0;
	double taper_y = 0.0;
	double taper = 0.0;
	double l = 0.0;
	double m = 0.0;
	int grid_index = 0;
	for(int row_index = 0; row_index < grid_size; ++row_index)
	{
		nu_y = fabs((row_index - grid_half_size) / grid_half_size);
		taper_y = calc_spheroidal_sample(nu_y);
		m = pow((row_index - grid_half_size) * cell_size, 2.0);

		for(int col_index = 0; col_index < grid_size; ++col_index)
		{
			nu_x = fabs((col_index - grid_half_size) / grid_half_size);
			taper = taper_y * calc_spheroidal_sample(nu_x);
			grid_index = row_index * grid_size + col_index;

			
			if(fabs(taper) > (1E-10))
			{
				l = pow((col_index - grid_half_size) * cell_size, 2.0);
				grid[grid_index].real /= (taper * sqrt(1.0 - l - m));
			}
			else
				grid[grid_index].real = 0.0;
		}
	}
}

void unit_test_init_config(Config *config)
{
	config->grid_size 							= 18000;
	config->right_ascension 					= true;
	config->cell_size 							= 6.39708380288950e-6;
	config->frequency_hz 						= 100e6;
	config->oversampling 						= 4;
	config->uv_scale 							= config->grid_size * config->cell_size;
	config->grid_real_source_file 				= "../unit_test_data/ModelImage_18000.csv";
	config->kernel_support_file         		= "../unit_test_data/kernels/el82-el70_kernel_supports.csv";
	config->kernel_real_source_file 			= "../unit_test_data/kernels/el82-el70_kernel_real.csv";
	config->kernel_imag_source_file 			= "../unit_test_data/kernels/el82-el70_kernel_imag.csv";
	config->visibility_source_file 				= "../unit_test_data/el82-70_unit_test.csv";
	config->visibility_dest_file 				= "../el82-70_unit_test_output.csv";
	config->gpu_max_threads_per_block 			= 1024;
	config->gpu_max_threads_per_block_dimension = 32;
	config->time_degridding 					= true;
	config->conv_correction_and_fft 			= true;
	config->conv_correction_performed 			= false;
	config->force_zero_w_term 					= false;
	config->num_wproj_kernels 					= 339;
	config->max_w 								= 7083.386050;
	config->w_scale 							= pow(config->num_wproj_kernels - 1, 2.0) / config->max_w;
}

double unit_test_output_visibilities(Config *config, Visibility *vis_uvw, Complex *vis_intensities)
{
	// Attempt to open visibility source file
	FILE *vis_file = fopen(config->visibility_source_file, "r");
	if(vis_file == NULL)
	{
		printf("Unable to open visibility file...\n");
		return DBL_MAX; // unsuccessfully loaded data
	}
	
	// Configure number of visibilities from file
	int num_vis = 0;
	fscanf(vis_file, "%d", &num_vis);
	config->num_visibilities = num_vis;

	// Load visibility uvw coordinates into memory
	double vis_u = 0.0;
	double vis_v = 0.0;
	double vis_w = 0.0;
	double vis_real = 0.0;
	double vis_imag = 0.0;
	double vis_weight = 0.0;

	double max_error = 0.0;
	for(int vis_index = 0; vis_index < num_vis; ++vis_index)
	{
		// Discard vis(real), vis(imag), and weighting (for now)
		fscanf(vis_file, "%lf %lf %lf %lf %lf %lf\n", &vis_u, &vis_v,
			&vis_w, &vis_real, &vis_imag, &vis_weight);
		
		double current_diff = sqrt(pow(vis_real - vis_intensities[vis_index].real, 2.0)
	  		+ pow(vis_imag - vis_intensities[vis_index].imag, 2.0));

		if(current_diff > max_error)
			max_error = current_diff;
	}
	// Clean up
	fclose(vis_file);
	return max_error;
}
