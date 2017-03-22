/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This header file lists a private kernel function call for testing.
 */

#ifndef BAYER2RGB_KERNEL_H
#define BAYER2RGB_KERNEL_H

#include <cuda_runtime.h>

typedef void (*bayer_to_rgb_t)(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp, int2 r, int2 gr, int2 gb, int2 b);

/**
 * CUDA Kernel Device code for RGGB
 *
 * Computes the Bilear Interpolation of missing coloured pixel from Bayer pattern.
 * Output is RGB.
 */
__global__ void bayer_to_rgb(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp, int2 r, int2 gr, int2 gb, int2 b);

__global__ void bayer_ir_to_rgb(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp, int2 r, int2 ir, int2 g, int2 b);

__global__ void bayer_ir_to_thermal(uint8_t *in, uint8_t *out, uint32_t imgw,
		uint32_t imgh, uint8_t bpp, int2 r, int2 ir, int2 g, int2 b);

#endif
