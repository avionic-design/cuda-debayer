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

#include <linux/videodev2.h>

#include "bayer2rgb.h"
#include "bayer2rgb_kernel.h"

#define LEFT(x, y, imgw)	((x) - 1 + (y) * (imgw))
#define RIGHT(x, y, imgw)	((x) + 1 + (y) * (imgw))
#define TOP(x, y, imgw)		((x) + ((y) - 1) * (imgw))
#define BOT(x, y, imgw)		((x) + ((y) + 1) * (imgw))
#define TL(x, y, imgw)		((x) - 1 + ((y) - 1) * (imgw))
#define BL(x, y, imgw)		((x) - 1 + ((y) + 1) * (imgw))
#define TR(x, y, imgw)		((x) + 1 + ((y) - 1) * (imgw))
#define BR(x, y, imgw)		((x) + 1 + ((y) + 1) * (imgw))

#define PIX(in, x, y, imgw) \
	in[((x) + (y) * (imgw))]

#define INTERPOLATE_H(in, x, y, w) \
	(((uint32_t)in[LEFT(x, y, w)] + in[RIGHT(x, y, w)]) / 2)

#define INTERPOLATE_V(in, x, y, w) \
	(((uint32_t)in[TOP(x, y, w)] + in[BOT(x, y, w)]) / 2)

#define INTERPOLATE_HV(in, x, y, w) \
	(((uint32_t)in[LEFT(x, y, w)] + in[RIGHT(x, y, w)] + \
		in[TOP(x, y, w)] + in[BOT(x, y, w)]) / 4)

#define INTERPOLATE_X(in, x, y, w) \
	(((uint32_t)in[TL(x, y, w)] + in[BL(x, y, w)] + \
		in[TR(x, y, w)] + in[BR(x, y, w)]) / 4)

#define RED 0
#define GREEN 1
#define BLUE 2

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
 *
 * Each CUDA thread computes four pixels in a 2x2 square. Therefore no if
 * conditions are required, which slows the CUDA kernels massively.
 *
 * The first square starts with the pixel in position 1,1. Therefore the square
 * for each thread looks like this:
 *
 * B G
 * G R
 *
 * This approach saves one pixel lines at the edges of the image in contrast to
 * the first square at 2,2 with:
 *
 * R G
 * G B
 *
 */
__global__ void bayer_to_rgb(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp)
{
	int x = 2 * ((blockDim.x * blockIdx.x) + threadIdx.x) + 1;
	int y = 2 * ((blockDim.y * blockIdx.y) + threadIdx.y) + 1;
	int elemCols = imgw * bpp;

	if ((x + 2) < imgw && (x - 1) >= 0 && (y + 2) < imgh && (y - 1) >= 0) {
		/* red at red */
		out[(y + 1) * elemCols + (x + 1) * bpp + RED] =
				PIX(in, x + 1, y + 1, imgw);
		/* green at red */
		out[(y + 1) * elemCols + (x + 1) * bpp + GREEN] =
				INTERPOLATE_HV(in, x + 1, y + 1, imgw);
		/* blue at red */
		out[(y + 1) * elemCols + (x + 1) * bpp + BLUE] =
				INTERPOLATE_X(in, x + 1, y + 1, imgw);

		/* red at lower left green */
		out[(y + 1) * elemCols + x * bpp + RED] =
				INTERPOLATE_H(in, x, y + 1, imgw);
		/* green at lower left green */
		out[(y + 1) * elemCols + x * bpp + GREEN] =
				PIX(in, x, y + 1, imgw);
		/* blue at lower left green */
		out[(y + 1) * elemCols + x * bpp + BLUE] =
				INTERPOLATE_V(in, x, y + 1, imgw);

		/* red at upper right green */
		out[y * elemCols + (x + 1) * bpp + RED] =
				INTERPOLATE_V(in, x + 1, y, imgw);
		/* green at upper right green */
		out[y * elemCols + (x + 1) * bpp + GREEN] =
				PIX(in, x + 1, y, imgw);
		/* blue at upper right green */
		out[y * elemCols + (x + 1) * bpp + BLUE] =
				INTERPOLATE_H(in, x + 1, y, imgw);

		/* red at blue */
		out[y * elemCols + x * bpp + RED] =
				INTERPOLATE_X(in, x, y, imgw);
		/* green at blue */
		out[y * elemCols + x * bpp + GREEN] =
				INTERPOLATE_HV(in, x, y, imgw);
		/* blue at blue */
		out[y * elemCols + x * bpp + BLUE] =
				PIX(in, x, y, imgw);

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

	ret_val = cudaMemcpyAsync(gpu_vars->d_input[gpu_vars->cnt],
			p, gpu_vars->width * gpu_vars->height *
			sizeof(uint8_t), cudaMemcpyHostToDevice,
			gpu_vars->streams[gpu_vars->cnt]);
	if (ret_val != cudaSuccess) {
		fprintf(stderr, "Host to Device %d, %s\n", gpu_vars->cnt,
				cudaGetErrorString(ret_val));
		return ret_val;
	}

	bayer_to_rgb<<<gpu_vars->blocks_p_grid,
			gpu_vars->threads_p_block, 0,
			gpu_vars->streams[gpu_vars->cnt]
		>>>(gpu_vars->d_input[gpu_vars->cnt],
			gpu_vars->d_bilinear[gpu_vars->cnt],
			gpu_vars->width, gpu_vars->height, gpu_vars->bpp);

	if (get_dev_ptr) {
		*output = (uint8_t *)gpu_vars->d_bilinear[gpu_vars->cnt];
	} else {
		ret_val = cudaMemcpyAsync(*output,
			gpu_vars->d_bilinear[gpu_vars->cnt],
			gpu_vars->width * gpu_vars->height * sizeof(uint8_t) *
			gpu_vars->bpp, cudaMemcpyDeviceToHost,
			gpu_vars->streams[gpu_vars->cnt]);
		if (ret_val != cudaSuccess) {
			fprintf(stderr, "Device to Host %d, %s\n",
					gpu_vars->cnt,
					cudaGetErrorString(ret_val));
			return ret_val;
		}
		ret_val = cudaStreamSynchronize(
				gpu_vars->streams[gpu_vars->cnt]);
		if (ret_val != cudaSuccess) {
			fprintf(stderr, "device synchronize\n");
			return ret_val;
		}
	}

	*stream = gpu_vars->streams[gpu_vars->cnt];

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
		uint32_t height, uint8_t bpp, uint32_t format)
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

	switch (format) {
	case V4L2_PIX_FMT_SRGGB8:
		break;
	default:
		fprintf(stderr, "unsupported pixel format\n");
		ret_val = cudaErrorInvalidValue;
		goto cleanup;
	}

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
