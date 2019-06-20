
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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DEGRIDDER_H_
#define DEGRIDDER_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cufft.h>

	#define C 299792458.0

	#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

	#define CUFFT_SAFE_CALL(err) cufft_safe_call(err, __FILE__, __LINE__)

	typedef struct Config {
		int grid_size;
		double cell_size;
		bool right_ascension;
		bool force_zero_w_term;
		double frequency_hz;
		int kernel_size;
		int oversampling;
		double uv_scale;
		int num_visibilities;
		char *grid_real_source_file;
		char *grid_imag_source_file;
		char *kernel_real_source_file;
		char *kernel_imag_source_file;
		char *kernel_support_file;
		char *visibility_source_file;
		char *visibility_dest_file;
		int gpu_max_threads_per_block;
		int gpu_max_threads_per_block_dimension;
		bool time_degridding;
		bool conv_correction_and_fft;
		bool conv_correction_performed;
		int num_wproj_kernels;
		double max_w;
		double w_scale;
	} Config;

	typedef struct Visibility {
		double u;
		double v;
		double w;
	} Visibility;

	typedef struct Complex {
		double real;
		double imag;
	} Complex;

	void init_config(Config *config);

	bool load_grid(Config *config, Complex *grid);

	bool load_visibilities(Config *config, Visibility **vis_uvw, Complex **vis_intensities);

	void save_visibilities(Config *config, Visibility *vis_uvw, Complex *vis_intensity);

	void execute_degridding(Config *config, Complex *grid, 
		Visibility *vis_uvw, Complex *vis_intensities, int num_visibilities, double2 *prolate,
		Complex *kernel, int2 *kernel_supports, int num_kernel_samples);

	__global__ void execute_convolution_correction(double2 *grid, const double2 *prolate, const int grid_size);

	__global__ void degridding(const double2 *grid, const double2 *kernel, const int2 *supports,
		const double3 *vis_uvw, double2 *vis, const int num_vis, const int oversampling,
		const int grid_size, const double uv_scale, const double w_scale);

	__device__ double2 complex_mult(const double2 z1, const double2 z2);

	bool load_kernel(Config *config, Complex *kernel, int2 *kernel_supports);

	int read_kernel_supports(Config *config, int2 *kernel_supports);

	double calc_spheroidal_sample(double nu);

	void execute_convolution_correction_cpu(Complex *grid, double grid_size, double cell_size);

	void execute_CUDA_FFT(Config *config, double2 *grid);

	void create_1D_half_prolate(double2 *prolate, int grid_size, double cell_size);

	__global__ void fftshift_2D(double2 *grid, const int width);

	void clean_up(Complex **grid, Visibility **visibilities, Complex **vis_intensities, Complex **kernel, int2 **kernel_supports, double2 **prolate);

	static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

	static void cufft_safe_call(cufftResult err, const char *file, const int line);

	static const char* cuda_get_error_enum(cufftResult error);

	void unit_test_init_config(Config *config);

	double unit_test_output_visibilities(Config *config, Visibility *vis_uvw, Complex *vis_intensities);

	double unit_test_gpu_convolution_correction(Complex *grid, int grid_size, double cell_size);


#endif /* DEGRIDDER_H_ */

#ifdef __cplusplus
}
#endif
