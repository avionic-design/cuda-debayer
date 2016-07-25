/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file contains the CUDA kernel with functions to initialise all needed
 * parameters for kernel Launch. Also the destruction of generated parameter is
 * included.
 *
 * Compute Capability 3.0 or higher required
 */

#include "bayer2rgb.h"

#define PIX(x, y, imgw)		((x) + (y) * (imgw))
#define LEFT(x, y, imgw)	((x) - 1 + (y) * (imgw))
#define RIGHT(x, y, imgw)	((x) + 1 + (y) * (imgw))
#define TOP(x, y, imgw)		((x) + ((y) - 1) * (imgw))
#define BOT(x, y, imgw)		((x) + ((y) + 1) * (imgw))
#define TL(x, y, imgw)		((x) - 1 + ((y) - 1) * (imgw))
#define BL(x, y, imgw)		((x) - 1 + ((y) + 1) * (imgw))
#define TR(x, y, imgw)		((x) + 1 + ((y) - 1) * (imgw))
#define BR(x, y, imgw)		((x) + 1 + ((y) + 1) * (imgw))

struct cuda_vars {
	cudaArray *data[2];

	dim3 threads_p_block;
	dim3 blocks_p_grid;

	uint8_t *d_bilinear[2];
	uint8_t *d_input[2];

	uint32_t width;
	uint32_t height;

	cudaStream_t streams[2];

	uint8_t cnt;
	uint8_t bpp;
};

/**
 * CUDA Kernel Device code for RGGB
 *
 * Computes the Bilear Interpolation of missing coloured pixel from Bayer pattern.
 * Output is RGB.
 */
__global__ void bayer_to_rgb(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp)
{
	int x = 2 * ((blockDim.x * blockIdx.x) + threadIdx.x);
	int y = 2 * ((blockDim.y * blockIdx.y) + threadIdx.y);
	int elemCols = imgw * bpp;

	if ((x + 2) < imgw && (x - 1) >= 0 && (y + 2) < imgh && (y - 1) >= 0) {
		/* red at red */
		out[y * elemCols + x * bpp] = in[PIX(x, y, imgw)];
		/* green at red */
		out[y * elemCols + x * bpp + 1] =
				((uint32_t)in[TOP(x, y, imgw)] +
				in[BOT(x, y, imgw)] +
				in[LEFT(x, y, imgw)] +
				in[RIGHT(x, y, imgw)]) / 4;
		/* blue at red */
		out[y * elemCols + x * bpp + 2] =
				((uint32_t)in[TL(x, y, imgw)] +
				in[TR(x, y, imgw)] +
				in[BL(x, y, imgw)] +
				in[BR(x, y, imgw)]) / 4;

		/* red at upper right green */
		out[y * elemCols + (x + 1) * bpp] =
				((uint32_t)in[LEFT(x + 1, y, imgw)] +
				in[RIGHT(x + 1, y, imgw)]) / 2;
		/* green at upper right green */
		out[y * elemCols + (x + 1) * bpp + 1] =
				in[PIX(x + 1, y, imgw)];
		/* blue at upper right green */
		out[y * elemCols + (x + 1) * bpp + 2] =
				((uint32_t)in[TOP(x + 1, y, imgw)] +
				in[BOT(x + 1, y, imgw)]) / 2;

		/* red at lower left green */
		out[(y + 1) * elemCols + x * bpp] =
				((uint32_t)in[TOP(x, y + 1, imgw)] +
				in[BOT(x, y + 1, imgw)]) / 2;
		/* green at lower left green */
		out[(y + 1) * elemCols + x * bpp + 1] =
				in[PIX(x, y + 1, imgw)];
		/* blue at lower left green */
		out[(y + 1) * elemCols + x * bpp + 2] =
				((uint32_t)in[LEFT(x, y + 1, imgw)] +
				in[RIGHT(x, y + 1, imgw)]) / 2;

		/* red at blue */
		out[(y + 1) * elemCols + (x + 1) * bpp] =
				((uint32_t)in[TL(x + 1, y + 1, imgw)] +
				in[TR(x + 1, y + 1, imgw)] +
				in[BL(x + 1, y + 1, imgw)] +
				in[BR(x + 1, y + 1, imgw)]) / 4;
		/* green at blue */
		out[(y + 1) * elemCols + (x + 1) * bpp + 1] =
				((uint32_t)in[TOP(x + 1, y + 1, imgw)] +
				in[BOT(x + 1, y + 1, imgw)] +
				in[LEFT(x + 1, y + 1, imgw)] +
				in[RIGHT(x + 1, y + 1, imgw)]) / 4;
		/* blue at blue */
		out[(y + 1) * elemCols + (x + 1) * bpp + 2] =
				in[PIX(x + 1, y + 1, imgw)];

		if (bpp == 4) {
			out[y * elemCols + x * bpp + 3] = 255;
			out[y * elemCols + (x + 1) * bpp + 3] = 255;
			out[(y + 1) * elemCols + x * bpp + 3] = 255;
			out[(y + 1) * elemCols + (x + 1) * bpp + 3] = 255;
		}
	}
}

cudaError_t bayer2rgb_process(struct cuda_vars *gpu_vars, const void *p,
		uint8_t **output, cudaStream_t *stream, bool get_dev_ptr)
{
	cudaError_t ret_val;

	if (gpu_vars == NULL)
		return cudaErrorInitializationError;

	ret_val = cudaMemcpyAsync(gpu_vars->d_input[(gpu_vars->cnt % 2)],
			p, gpu_vars->width * gpu_vars->height *
			sizeof(uint8_t), cudaMemcpyHostToDevice,
			gpu_vars->streams[gpu_vars->cnt % 2]);
	if (ret_val != cudaSuccess) {
		fprintf(stderr, "Host to Device %d, %s\n", gpu_vars->cnt % 2,
				cudaGetErrorString(ret_val));
		return ret_val;
	}

	bayer_to_rgb<<<gpu_vars->blocks_p_grid,
			gpu_vars->threads_p_block, 0,
			gpu_vars->streams[(gpu_vars->cnt % 2)]
		>>>(gpu_vars->d_input[(gpu_vars->cnt % 2)],
			gpu_vars->d_bilinear[(gpu_vars->cnt % 2)],
			gpu_vars->width, gpu_vars->height, gpu_vars->bpp);

	if (get_dev_ptr) {
		*output = (uint8_t *)gpu_vars->d_bilinear[(gpu_vars->cnt % 2)];
	} else {
		ret_val = cudaMemcpyAsync(*output,
			gpu_vars->d_bilinear[gpu_vars->cnt % 2],
			gpu_vars->width * gpu_vars->height * sizeof(uint8_t) *
			gpu_vars->bpp, cudaMemcpyDeviceToHost,
			gpu_vars->streams[gpu_vars->cnt % 2]);
		if (ret_val != cudaSuccess) {
			fprintf(stderr, "Device to Host %d, %s\n",
					gpu_vars->cnt % 2,
					cudaGetErrorString(ret_val));
			return ret_val;
		}
	}

	ret_val = cudaStreamSynchronize(gpu_vars->streams[
			gpu_vars->cnt % 2]);
	if (ret_val != cudaSuccess) {
		fprintf(stderr, "device synchronize\n");
		return ret_val;
	}

	gpu_vars->cnt = (gpu_vars->cnt + 1) % 2;

	return cudaSuccess;
}

cudaError_t alloc_create_cuda_data(struct cuda_vars *gpu_vars, uint8_t cnt)
{
	cudaError_t ret_val = cudaSuccess;

	ret_val = cudaMalloc(&gpu_vars->d_input[cnt], gpu_vars->width *
			gpu_vars->height * sizeof(uint8_t));
	if (ret_val != cudaSuccess) {
		fprintf(stderr, "cudaMalloc d_bilinear %d, %s\n", cnt,
				cudaGetErrorString(ret_val));
		return ret_val;
	}

	ret_val = cudaMalloc(&gpu_vars->d_bilinear[cnt], gpu_vars->width *
			gpu_vars->height * sizeof(uint8_t) *
			gpu_vars->bpp);
	if (ret_val != cudaSuccess) {
		fprintf(stderr, "cudaMalloc d_bilinear %d, %s\n", cnt,
				cudaGetErrorString(ret_val));
		return ret_val;
	}

	ret_val = cudaStreamCreate(&gpu_vars->streams[cnt]);
	if (ret_val != cudaSuccess) {
		fprintf(stderr, "cudaStreamCreate %d, %s\n", cnt,
				cudaGetErrorString(ret_val));
		return ret_val;
	}

	return ret_val;
}

cudaError_t bayer2rgb_init(struct cuda_vars **gpu_vars_p, uint32_t width,
		uint32_t height, uint8_t bpp)
{
	struct cuda_vars *gpu_vars;
	cudaError_t ret_val;
	int i;

	if (gpu_vars_p == NULL)
		return cudaErrorInitializationError;

	gpu_vars = (cuda_vars *) new(struct cuda_vars);
	if (!gpu_vars)
		return cudaErrorMemoryAllocation;

	gpu_vars->width = width;
	gpu_vars->height = height;
	gpu_vars->cnt = 0;
	gpu_vars->bpp = bpp;

	for (i = 0; i < 2; i++) {
		ret_val = alloc_create_cuda_data(gpu_vars, i);
		if (ret_val != cudaSuccess)
			goto cleanup;
	}

	gpu_vars->threads_p_block = dim3(32, 32);
	gpu_vars->blocks_p_grid.x = (gpu_vars->width / 2 +
			gpu_vars->threads_p_block.x - 1) /
			gpu_vars->threads_p_block.x;
	gpu_vars->blocks_p_grid.y = (gpu_vars->height / 2 +
			gpu_vars->threads_p_block.y - 1) /
			gpu_vars->threads_p_block.y;

	*gpu_vars_p = gpu_vars;

	return cudaSuccess;

cleanup:
	bayer2rgb_free(gpu_vars);

	return ret_val;
}

void free_cuda_data(struct cuda_vars *gpu_vars, uint8_t cnt)
{
	if (gpu_vars->d_input[cnt])
		cudaFree(gpu_vars->d_input[cnt]);
	if (gpu_vars->d_bilinear[cnt])
		cudaFree(gpu_vars->d_bilinear[cnt]);
	cudaStreamDestroy(gpu_vars->streams[cnt]);
}

cudaError_t bayer2rgb_free(struct cuda_vars *gpu_vars)
{
	int i;

	for (i = 0; i < 2; i++) {
		free_cuda_data(gpu_vars, i);
	}

	free(gpu_vars);

	return cudaSuccess;
}
