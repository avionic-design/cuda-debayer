/*
 * Copyright (C) 2016 Avionic Design GmbH
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <errno.h>
#include <getopt.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

static const char short_options[] = "ho";

static const struct option long_options[] = {
	{"help",	no_argument,		NULL, 'h'},
	{"output",	no_argument,		NULL, 'o'},
	{0, 0, 0, 0 }
};

static void usage(FILE *fp, const char *argv)
{
	fprintf(fp,
		"Usage: %s [options]\n\n"
		"Options:\n"
		"-h | --help          Print this message\n"
		"-o | --output        Outputs stream to screen\n"
		"",
		argv);
}

int main(int argc, char **argv)
{
	cv::Mat image = cv::Mat::zeros(1, 1, CV_8UC3);
	const std::string window = "Display";
	bool displayed = false;

	for (;;) {
		int idx;
		int c;

		c = getopt_long(argc, argv, short_options, long_options, &idx);
		if (c == -1)
			break;
		switch (c) {
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

	cv::namedWindow(window, CV_WINDOW_NORMAL);

	while (true) {
		cv::imshow(window, image);
		cv::waitKey(1);
	}

	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
