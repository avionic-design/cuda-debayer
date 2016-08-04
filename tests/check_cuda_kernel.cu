/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file tests the bayer2rgb kernel.
 */

#include <cuda_runtime.h>

#include <check.h>
#include <stdlib.h>
#include <stdio.h>

#include "bayer2rgb_kernel.h"
#include "cuda_kernel_test_data.h"

#define PIX(x, y, w)	((x) + (y) * (w))
#define LEFT(x, y, w)	((x) - 1 + (y) * (w))
#define RIGHT(x, y, w)	((x) + 1 + (y) * (w))
#define TOP(x, y, w)	((x) + ((y) - 1) * (w))
#define BOT(x, y, w)	((x) + ((y) + 1) * (w))
#define TL(x, y, w)	((x) - 1 + ((y) - 1) * (w))
#define BL(x, y, w)	((x) - 1 + ((y) + 1) * (w))
#define TR(x, y, w)	((x) + 1 + ((y) - 1) * (w))
#define BR(x, y, w)	((x) + 1 + ((y) + 1) * (w))

START_TEST(test_kernel)
{
	cudaError_t ret_val = cudaSuccess;
	uint8_t *gpu_output;
	uint8_t *d_output;
	uint8_t *d_input;
	uint8_t bpp = 3;
	uint8_t *data;
	uint32_t height = 16;
	uint32_t width = 16;
	int i, j;

	gpu_output = (uint8_t *) malloc(width * height * sizeof(uint8_t) * bpp);

	data = (uint8_t *)&input_data;

	ret_val = cudaMalloc(&d_input, width * height * sizeof(uint8_t));
	if (ret_val != cudaSuccess)
		ck_abort_msg("cudaMalloc input failure");

	ret_val = cudaMalloc(&d_output, width * height * sizeof(uint8_t) * bpp);
	if (ret_val != cudaSuccess)
		ck_abort_msg("cudaMalloc output failure");

	ret_val = cudaMemcpy(d_input, data, width * height *
			sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (ret_val != cudaSuccess)
		ck_abort_msg("Host to Device copy failure");

	bayer_to_rgb<<<dim3(width, height), dim3(1,1)>>>(d_input, d_output,
			width, height, bpp);

	ret_val = cudaMemcpy(gpu_output, d_output, width * height *
			sizeof(uint8_t) * bpp, cudaMemcpyDeviceToHost);
	if (ret_val != cudaSuccess)
		ck_abort_msg("Device to Host copy failure");

	cudaDeviceSynchronize();

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			ck_assert_uint_eq(gpu_output[j * width * bpp + i * bpp],
					output_data_red[j * width + i]);
			ck_assert_uint_eq(gpu_output[j * width * bpp + i * bpp + 1],
					output_data_green[j * width + i]);
			ck_assert_uint_eq(gpu_output[j * width * bpp + i * bpp + 2],
					output_data_blue[j * width + i]);
		}
	}
}
END_TEST

Suite* bayer2rgb_suite(void)
{
	TCase *tc_bayer2rgb = tcase_create("bayer2rgb");
	Suite *s = suite_create("bayer2rgb");

	tcase_add_test(tc_bayer2rgb, test_kernel);
	suite_add_tcase(s, tc_bayer2rgb);

	return s;
}

int main()
{
	Suite *s = bayer2rgb_suite();
	SRunner *sr = srunner_create(s);
	int number_failed;

	srunner_run_all(sr, CK_ENV);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);

	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
