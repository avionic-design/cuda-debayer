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
#include <string>
#include <errno.h>
#include <getopt.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "camera_device.h"

static const char short_options[] = "d:e:g:ho";

static const struct option long_options[] = {
	{"device",	required_argument,	NULL, 'd'},
	{"exposure",	required_argument,	NULL, 'e'},
	{"gain",	required_argument,	NULL, 'g'},
	{"help",	no_argument,		NULL, 'h'},
	{"output",	no_argument,		NULL, 'o'},
	{0, 0, 0, 0 }
};

static void usage(FILE *fp, const char *argv)
{
	fprintf(fp,
		"Usage: %s [options]\n\n"
		"Options:\n"
		"-d | --device        Video device name [default:/dev/video0]\n"
		"-e | --exposure      Set exposure time\n"
		"-g | --gain          Set analog gain\n"
		"-h | --help          Print this message\n"
		"-o | --output        Outputs stream to screen\n"
		"",
		argv);
}

int main(int argc, char **argv)
{
	cv::Mat image = cv::Mat::zeros(1, 1, CV_8UC3);
	const std::string window = "Display";
	std::string dev_name = "/dev/video0";
	struct camera_vars *cam_vars;
	bool displayed = false;
	int exposure = -1;
	int ret_val = 0;
	long int l_int;
	int gain = -1;

	for (;;) {
		int idx;
		int c;

		c = getopt_long(argc, argv, short_options, long_options, &idx);
		if (c == -1)
			break;
		switch (c) {
		case 'd':
			dev_name = optarg;
			if (access(dev_name.c_str(), F_OK) == -1) {
				printf("device %s does not exist\n",
						dev_name.c_str());
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
			displayed = true;
			break;

		default:
			usage(stderr, argv[0]);
			return EXIT_FAILURE;
		}
	}

	if (!displayed) {
		printf("not displayed\n");
		return EXIT_SUCCESS;
	}

	ret_val = camera_device_init(&cam_vars, dev_name, exposure, gain);
	if (ret_val != 0)
		goto cleanup;

	image = cv::Mat(camera_device_get_height(cam_vars),
			camera_device_get_width(cam_vars), CV_8UC3);

	cv::namedWindow(window, CV_WINDOW_NORMAL);
	cv::resizeWindow(window,
			camera_device_get_width(cam_vars),
			camera_device_get_height(cam_vars));

	while (true) {
		ret_val = camera_device_read_frame(cam_vars, &(image.data));
		if (ret_val == -EAGAIN)
			continue;

		if (ret_val == CANCEL)
			break;

		if (ret_val != 0)
			goto cleanup;

		cv::imshow(window, image);
		cv::waitKey(1);
	}

	ret_val = EXIT_SUCCESS;

cleanup:
	if (cam_vars != NULL)
		camera_device_done(cam_vars);

	if (gpu_vars != NULL) {
		ret_cuda = bayer2rgb_done(gpu_vars);
		if (ret_cuda != cudaSuccess)
			return -EPERM;
	}

	cv::destroyAllWindows();

	return ret_val;
}
