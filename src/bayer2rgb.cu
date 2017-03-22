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
#include "bayer2rgb_kernel.h"
#include "camera_device.h"

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

	bayer_to_rgb_t kernel;
	dim3 threads_p_block;
	dim3 blocks_p_grid;

	int2 pos_r;
	int2 pos_gr;
	int2 pos_gb;
	int2 pos_b;

	uint8_t *d_bilinear[2];
	uint8_t *d_input[2];

	uint32_t width;
	uint32_t height;

	cudaStream_t streams[2];

	uint8_t cnt;
	uint8_t bpp;
};

/**
 * CUDA Kernel Device code for bayer to RGB
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
 * To support other formats than RGGB we also pass the position of each color
 * channel in the 2x2 block. In the above case we get B at 0,0, Gb at 1,0,
 * Gr at 0,1 and R at 1,1.
 */
__global__ void bayer_to_rgb(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp, int2 r, int2 gr, int2 gb, int2 b)
{
	int x = 2 * ((blockDim.x * blockIdx.x) + threadIdx.x) + 1;
	int y = 2 * ((blockDim.y * blockIdx.y) + threadIdx.y) + 1;
	int elemCols = imgw * bpp;

	if ((x + 2) < imgw && (x - 1) >= 0 && (y + 2) < imgh && (y - 1) >= 0) {
		/* Red */
		out[(y + r.y) * elemCols + (x + r.x) * bpp + RED] =
				PIX(in, x + r.x, y + r.y, imgw);
		out[(y + r.y) * elemCols + (x + r.x) * bpp + GREEN] =
				INTERPOLATE_HV(in, x + r.x, y + r.y, imgw);
		out[(y + r.y) * elemCols + (x + r.x) * bpp + BLUE] =
				INTERPOLATE_X(in, x + r.x, y + r.y, imgw);

		/* Green on a red line */
		out[(y + gr.y) * elemCols + (x + gr.x) * bpp + RED] =
				INTERPOLATE_H(in, x + gr.x, y + gr.y, imgw);
		out[(y + gr.y) * elemCols + (x + gr.x) * bpp + GREEN] =
				PIX(in, x + gr.x, y + gr.y, imgw);
		out[(y + gr.y) * elemCols + (x + gr.x) * bpp + BLUE] =
				INTERPOLATE_V(in, x + gr.x, y + gr.y, imgw);

		/* Green on a blue line */
		out[(y + gb.y) * elemCols + (x + gb.x) * bpp + RED] =
				INTERPOLATE_V(in, x + gb.x, y + gb.y, imgw);
		out[(y + gb.y) * elemCols + (x + gb.x) * bpp + GREEN] =
				PIX(in, x + gb.x, y + gb.y, imgw);
		out[(y + gb.y) * elemCols + (x + gb.x) * bpp + BLUE] =
				INTERPOLATE_H(in, x + gb.x, y + gb.y, imgw);

		/* Blue */
		out[(y + b.y) * elemCols + (x + b.x) * bpp + RED] =
				INTERPOLATE_X(in, x + b.x, y + b.y, imgw);
		out[(y + b.y) * elemCols + (x + b.x) * bpp + GREEN] =
				INTERPOLATE_HV(in, x + b.x, y + b.y, imgw);
		out[(y + b.y) * elemCols + (x + b.x) * bpp + BLUE] =
				PIX(in, x + b.x, y + b.y, imgw);

		if (bpp == 4) {
			out[y * elemCols + x * bpp + 3] = 255;
			out[y * elemCols + (x + 1) * bpp + 3] = 255;
			out[(y + 1) * elemCols + x * bpp + 3] = 255;
			out[(y + 1) * elemCols + (x + 1) * bpp + 3] = 255;
		}
	}
}

/**
 * CUDA kernel for bayer with infrared to RGB
 *
 * This handle the conversion for bayer pattern that have infrared instead
 * of a second green pixel. The pattern look like this:
 *
 * R I or I R or B G or G B
 * G B    B G    I R    R I
 *
 */
__global__ void bayer_ir_to_rgb(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp, int2 r, int2 ir, int2 g, int2 b)
{
	int x = 2 * ((blockDim.x * blockIdx.x) + threadIdx.x) + 1;
	int y = 2 * ((blockDim.y * blockIdx.y) + threadIdx.y) + 1;
	int elemCols = imgw * bpp;

	if ((x + 2) < imgw && (x - 1) >= 0 && (y + 2) < imgh && (y - 1) >= 0) {
		/* Red */
		out[(y + r.y) * elemCols + (x + r.x) * bpp + RED] =
				PIX(in, x + r.x, y + r.y, imgw);
		out[(y + r.y) * elemCols + (x + r.x) * bpp + GREEN] =
				INTERPOLATE_V(in, x + r.x, y + r.y, imgw);
		out[(y + r.y) * elemCols + (x + r.x) * bpp + BLUE] =
				INTERPOLATE_X(in, x + r.x, y + r.y, imgw);

		/* Infrared */
		out[(y + ir.y) * elemCols + (x + ir.x) * bpp + RED] =
				INTERPOLATE_H(in, x + ir.x, y + ir.y, imgw);
		out[(y + ir.y) * elemCols + (x + ir.x) * bpp + GREEN] =
				INTERPOLATE_X(in, x + ir.x, y + ir.y, imgw);
		out[(y + ir.y) * elemCols + (x + ir.x) * bpp + BLUE] =
				INTERPOLATE_V(in, x + ir.x, y + ir.y, imgw);

		/* Green */
		out[(y + g.y) * elemCols + (x + g.x) * bpp + RED] =
				INTERPOLATE_V(in, x + g.x, y + g.y, imgw);
		out[(y + g.y) * elemCols + (x + g.x) * bpp + GREEN] =
				PIX(in, x + g.x, y + g.y, imgw);
		out[(y + g.y) * elemCols + (x + g.x) * bpp + BLUE] =
				INTERPOLATE_H(in, x + g.x, y + g.y, imgw);

		/* Blue */
		out[(y + b.y) * elemCols + (x + b.x) * bpp + RED] =
				INTERPOLATE_X(in, x + b.x, y + b.y, imgw);
		out[(y + b.y) * elemCols + (x + b.x) * bpp + GREEN] =
				INTERPOLATE_H(in, x + b.x, y + b.y, imgw);
		out[(y + b.y) * elemCols + (x + b.x) * bpp + BLUE] =
				PIX(in, x + b.x, y + b.y, imgw);

		if (bpp == 4) {
			out[y * elemCols + x * bpp + 3] = 255;
			out[y * elemCols + (x + 1) * bpp + 3] = 255;
			out[(y + 1) * elemCols + x * bpp + 3] = 255;
			out[(y + 1) * elemCols + (x + 1) * bpp + 3] = 255;
		}
	}
}

__device__ void ir_to_thermal(uint8_t *out, uint8_t h)
{
	int region, remainder;

	/* To convert the IR value to a thermogram we use a modified HSV
	 * color range. HSV is red, yellow, green, cyan, blue, purple but
	 * we want the reverse order starting with blue to get blue, cyan,
	 * green, yellow, red, purple. Furthermore we want a clear cut between
	 * the hot and cold regions so we make the purple-blue region to
	 * purple-white instead.
	 *
	 * The -1 offset is to have the 0 point starts at the end of the
	 * cyan-blue region, otherwise 0 would be white instead of blue.
	 */
	h = ((255 - h) + (43 * 4 - 1)) % 255;

	region = h / 43;
	remainder = (h - (region * 43)) * 6;

	switch (region) {
	case 0: /* red to yellow */
		out[RED] = 255;
		out[GREEN] = remainder;
		out[BLUE] = 0;
		break;
	case 1: /* yellow to green */
		out[RED] = 255 - remainder;
		out[GREEN] = 255;
		out[BLUE] = 0;
		break;
	case 2: /* green to cyan */
		out[RED] = 0;
		out[GREEN] = 255;
		out[BLUE] = remainder;
		break;
	case 3: /* cyan to blue */
		out[RED] = 0;
		out[GREEN] = 255 - remainder;
		out[BLUE] = 255;
		break;
	case 4: /* white to purple (instead of blue to purple) */
		out[RED] = 255;
		out[GREEN] = 255 - remainder;
		out[BLUE] = 255;
		break;
	case 5: /* purple to red */
		out[RED] = 255;
		out[GREEN] = 0;
		out[BLUE] = 255 - remainder;
		break;
	}
}

/**
 * CUDA kernel for bayer with infrared to a thermogram
 *
 * This kernel convert the IR channel to a false-color image.
 *
 */
__global__ void bayer_ir_to_thermal(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp, int2 r, int2 ir, int2 g, int2 b)
{
	int x = 2 * ((blockDim.x * blockIdx.x) + threadIdx.x) + 1;
	int y = 2 * ((blockDim.y * blockIdx.y) + threadIdx.y) + 1;
	int elemCols = imgw * bpp;

	if ((x + 2) < imgw && (x - 1) >= 0 && (y + 2) < imgh && (y - 1) >= 0) {
		/* Red */
		ir_to_thermal(&out[(y + r.y) * elemCols + (x + r.x) * bpp],
			INTERPOLATE_H(in, x + r.x, y + r.y, imgw));

		/* Infrared */
		ir_to_thermal(&out[(y + ir.y) * elemCols + (x + ir.x) * bpp],
			PIX(in, x + ir.x, y + ir.y, imgw));

		/* Green */
		ir_to_thermal(&out[(y + g.y) * elemCols + (x + g.x) * bpp],
			INTERPOLATE_X(in, x + g.x, y + g.y, imgw));

		/* Blue */
		ir_to_thermal(&out[(y + b.y) * elemCols + (x + b.x) * bpp],
			INTERPOLATE_V(in, x + b.x, y + b.y, imgw));

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

	gpu_vars->kernel<<<gpu_vars->blocks_p_grid,
			gpu_vars->threads_p_block, 0,
			gpu_vars->streams[gpu_vars->cnt]
		>>>(gpu_vars->d_input[gpu_vars->cnt],
			gpu_vars->d_bilinear[gpu_vars->cnt],
			gpu_vars->width, gpu_vars->height, gpu_vars->bpp,
			gpu_vars->pos_r, gpu_vars->pos_gr,
			gpu_vars->pos_gb, gpu_vars->pos_b);

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
		uint32_t height, uint8_t bpp, uint32_t format, bool thermal)
{
	struct cuda_vars *gpu_vars;
	bayer_to_rgb_t ir_kernel;
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
	gpu_vars->kernel = bayer_to_rgb;

	if (thermal)
		ir_kernel = bayer_ir_to_thermal;
	else
		ir_kernel = bayer_ir_to_rgb;

	switch (format) {
	case V4L2_PIX_FMT_SBGIR8:
		gpu_vars->kernel = ir_kernel;
	case V4L2_PIX_FMT_SBGGR8:
		gpu_vars->pos_r = make_int2(0, 0);
		gpu_vars->pos_gr = make_int2(1, 0);
		gpu_vars->pos_gb = make_int2(0, 1);
		gpu_vars->pos_b = make_int2(1, 1);
		break;
	case V4L2_PIX_FMT_SGBRI8:
		gpu_vars->kernel = ir_kernel;
	case V4L2_PIX_FMT_SGBRG8:
		gpu_vars->pos_r = make_int2(1, 0);
		gpu_vars->pos_gr = make_int2(0, 0);
		gpu_vars->pos_gb = make_int2(1, 1);
		gpu_vars->pos_b = make_int2(0, 1);
		break;
	case V4L2_PIX_FMT_SIRBG8:
		gpu_vars->kernel = ir_kernel;
	case V4L2_PIX_FMT_SGRBG8:
		gpu_vars->pos_r = make_int2(0, 1);
		gpu_vars->pos_gr = make_int2(1, 1);
		gpu_vars->pos_gb = make_int2(0, 0);
		gpu_vars->pos_b = make_int2(1, 0);
		break;
	case V4L2_PIX_FMT_SRIGB8:
		gpu_vars->kernel = ir_kernel;
	case V4L2_PIX_FMT_SRGGB8:
		gpu_vars->pos_r = make_int2(1, 1);
		gpu_vars->pos_gr = make_int2(0, 1);
		gpu_vars->pos_gb = make_int2(1, 0);
		gpu_vars->pos_b = make_int2(0, 0);
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
