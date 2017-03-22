/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This header file lists all public functions for initialisation, running and
 * destruction of the CUDA kernel.
 */

#ifndef BAYER2RGB_H
#define BAYER2RGB_H

#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>

struct cuda_vars;

cudaError_t bayer2rgb_init(struct cuda_vars **gpu_vars, uint32_t width,
		uint32_t height, uint8_t bpp, uint32_t format, bool thermal);
cudaError_t bayer2rgb_free(struct cuda_vars *gpu_vars);

cudaError_t bayer2rgb_process(struct cuda_vars *gpu_vars, const void *p,
		uint8_t **output, cudaStream_t *stream, bool get_dev_ptr);

#endif
