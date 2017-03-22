/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file contains a main function call for running camera_device functions.
 * The camera_device.h file is required with a implementation file.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <unistd.h>
#include <time.h>

#include <string>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include<cuda_profiler_api.h>

#include "camera_device.h"
#include "bayer2rgb.h"

#ifdef HAVE_OPENGL
#include "gl_display.h"
#endif

#define MAX_DECREASE 20

enum display_type{NOT, OPENCV, OPENGL};

static const char short_options[] = "d:e:g:ho:s:t";

static const struct option long_options[] = {
	{"device",	required_argument,	NULL, 'd'},
	{"exposure",	required_argument,	NULL, 'e'},
	{"gain",	required_argument,	NULL, 'g'},
	{"help",	no_argument,		NULL, 'h'},
	{"output",	required_argument,	NULL, 'o'},
	{"scale",	required_argument,	NULL, 's'},
	{"thermal",	no_argument,		NULL, 't'},
	{0, 0, 0, 0 }
};

static void usage(FILE *fp, const char *argv)
{
	fprintf(fp,
		"Usage: %s [-h|--help] \n"
		"   or: %s [-d|--device=/dev/video0] [-e|--exposure=30000] \n"
		"		[-g|--gain=100] [-s|--scale=1] [-o|--output=opengl]\n"
		"		[-t|--thermal]\n\n"
		"Options:\n"
		"-d | --device        Video device name [default:/dev/video0]\n"
		"-e | --exposure      Set exposure time [1..199000; default: 30000]\n"
		"-g | --gain          Set analog gain   [0..244; default: 100]\n"
		"-h | --help          Print this message\n"
		"-o | --output        Outputs stream over OpenCV/OpenGL [opencv, opengl; default: opengl]\n"
		"-s | --scale         Decreasing size by factor [1..%i; default: 1]\n"
		"-t | --thermal       Display a thermogram if the sensor support IR\n"
		"",
		argv, argv, MAX_DECREASE);
}

int main(int argc, char **argv)
{
	const std::string window = "Display";
	const char *dev_name = "/dev/video0";
	struct camera_vars *cam_vars = NULL;
	cudaError_t ret_cuda = cudaSuccess;
	struct cuda_vars *gpu_vars = NULL;
	cudaStream_t stream = NULL;
	uint8_t displayed = NOT;
	int exposure = 30000;
	bool thermal = false;
	int ret_val = 0;
	uint8_t *output;
	long int l_int;
	uint8_t *frame;
	int gain = -1;

#ifdef HAVE_OPENCV
	cv::Mat image = cv::Mat::zeros(1, 1, CV_8UC4);
	cv::Mat o_image;

	displayed = OPENCV;
#endif

#ifdef HAVE_OPENGL
	struct gl_display_vars *gl_vars = NULL;
	struct timespec start;
	struct timespec stop;
	int nframes = 0;
	double elapsed;
	double fps;

	displayed = OPENGL;
#endif

#if defined(HAVE_OPENCV) || defined(HAVE_OPENGL)
	float scale = 1.0f;
#endif

	for (;;) {
		int idx;
		int c;

		c = getopt_long(argc, argv, short_options, long_options, &idx);
		if (c == -1)
			break;
		switch (c) {
		case 'd':
			dev_name = optarg;
			if (access(dev_name, F_OK) == -1) {
				printf("device %s does not exist\n",
						dev_name);
				return EXIT_FAILURE;
			}
			break;

		case 'e':
			l_int = strtol(optarg, NULL, 10);
			if (errno == ERANGE || l_int > INT_MAX ||
				l_int < INT_MIN) {
				printf("argument of -e out of range\n");
				return EXIT_FAILURE;
			}

			exposure = l_int;
			break;

		case 'g':
			l_int = strtol(optarg, NULL, 10);
			if (errno == ERANGE || l_int > INT_MAX ||
				l_int < INT_MIN) {
				printf("argument of -g out of range\n");
				return EXIT_FAILURE;
			}

			gain = l_int;
			break;

		case 'h':
			usage(stdout, argv[0]);
			return EXIT_SUCCESS;

		case 'o':
			if (strcmp(optarg, "opencv") == 0) {
#ifndef HAVE_OPENCV
				printf("No OpenCV support\n");
				return EXIT_FAILURE;
#endif
				displayed = OPENCV;
			} else if (strcmp(optarg, "opengl") == 0) {
#ifndef HAVE_OPENGL
				printf("No OpenGL support\n");
				return EXIT_FAILURE;
#endif
				displayed = OPENGL;
			} else {
#ifdef HAVE_OPENGL
				displayed = OPENGL;
#elif HAVE_OPENCV
				displayed = OPENCV;
#endif
			}
			break;

		case 's':
			l_int = strtoul(optarg, NULL, 10);
			if (errno == ERANGE || l_int > MAX_DECREASE ||
				l_int < 0) {
				printf("argument of -s out of range\n");
				return EXIT_FAILURE;
			} else if (l_int == 0) {
				printf("invalid argument for scale (-s)\n");
				return EXIT_FAILURE;
			}

#if defined(HAVE_OPENCV) || defined(HAVE_OPENGL)
			scale = 1.0f / l_int;
#else
			printf("scaling without displaying is not supported\n");
			return EXIT_FAILURE;
#endif
			break;

		case 't':
			thermal = true;
			break;
		default:
			usage(stderr, argv[0]);
			return EXIT_FAILURE;
		}
	}

	cudaDeviceReset();
	cudaProfilerStart();

	ret_val = camera_device_init(&cam_vars, dev_name, exposure, gain);
	if (ret_val != 0)
		goto cleanup;

	ret_cuda = bayer2rgb_init(&gpu_vars,
			camera_device_get_width(cam_vars),
			camera_device_get_height(cam_vars), 4,
			camera_device_get_pixelformat(cam_vars),
			thermal);
	if (ret_cuda != cudaSuccess) {
		ret_val = -EINVAL;
		goto cleanup;
	}

	if (displayed == OPENGL) {
#ifdef HAVE_OPENGL
		printf("displayed with OpenGL\n");
		gl_display_init(&gl_vars, camera_device_get_width(cam_vars),
				camera_device_get_height(cam_vars), scale, argc,
				argv);
#endif
	} else if (displayed == OPENCV) {
#ifdef HAVE_OPENCV
		printf("displayed with OpenCV\n");
			cv::namedWindow(window, CV_WINDOW_NORMAL);
		cv::resizeWindow(window,
				camera_device_get_width(cam_vars) * scale,
				camera_device_get_height(cam_vars) * scale);
		image = cv::Mat(camera_device_get_height(cam_vars),
				camera_device_get_width(cam_vars), CV_8UC4);
		output = image.data;
#endif
	}

	while (true) {
		ret_val = camera_device_read_frame(cam_vars, &frame);
		if (ret_val == -EAGAIN)
			continue;

		if (ret_val == CANCEL)
			break;

		if (ret_val != 0)
			goto cleanup;

		ret_cuda = bayer2rgb_process(gpu_vars, frame, &output,
				&stream, (displayed == OPENCV) ? false : true);
		if (ret_cuda != cudaSuccess) {
			ret_val = -EINVAL;
			goto cleanup;
		}

		if (displayed == OPENGL) {
#ifdef HAVE_OPENGL
			nframes++;

			clock_gettime(CLOCK_REALTIME, &stop);
			elapsed = ((double)stop.tv_sec - (double)start.tv_sec)
					+ ((double)stop.tv_nsec -
					(double)start.tv_nsec) / 1000000000L;
			if (elapsed >= 1.0f) {
				fps = ((double)nframes) / elapsed;
				gl_display_fps(gl_vars, fps);
				nframes = 0;
				clock_gettime(CLOCK_REALTIME, &start);
			}

			gl_display_show(gl_vars, (unsigned int *)output,
					stream);
#endif
		} else if (displayed == OPENCV) {
#ifdef HAVE_OPENCV
			cv::cvtColor(image, o_image, CV_RGBA2BGRA);
			cv::imshow(window, o_image);
			cv::waitKey(1);
#endif
		}
	}

	ret_val = EXIT_SUCCESS;

cleanup:
	if (cam_vars != NULL)
		camera_device_done(cam_vars);

	if (gpu_vars != NULL) {
		ret_cuda = bayer2rgb_free(gpu_vars);
		if (ret_cuda != cudaSuccess)
			return -EINVAL;
	}

#ifdef HAVE_OPENGL
	if (gl_vars != NULL)
		ret_val = gl_display_free(gl_vars);
#endif

#ifdef HAVE_OPENCV
	if (displayed == OPENCV) {
		image.release();
		o_image.release();
		cv::destroyAllWindows();
	}
#endif

	return ret_val;
}
