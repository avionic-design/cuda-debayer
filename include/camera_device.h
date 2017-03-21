/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This header file provides public functions for initalisation, running and
 * destruction of the v4l2 camera. The bayer2rgb.h file is required with a
 * implementation file.
 */

#ifndef CAMERA_DEVICE_H
#define CAMERA_DEVICE_H

#include <stdint.h>
#include <linux/videodev2.h>

#ifndef V4L2_PIX_FMT_SBGIR8
#define V4L2_PIX_FMT_SBGIR8  v4l2_fourcc('I', 'R', '8', '1') /*  8  BGBG.. IRIR.. */
#define V4L2_PIX_FMT_SGBRI8  v4l2_fourcc('G', 'B', 'R', 'I') /*  8  GBGB.. RIRI.. */
#define V4L2_PIX_FMT_SIRBG8  v4l2_fourcc('I', 'R', 'B', 'G') /*  8  IRIR.. BGBG.. */
#define V4L2_PIX_FMT_SRIGB8  v4l2_fourcc('R', 'I', 'G', 'B') /*  8  RIRI.. GBGB.. */
#endif

/*
 * exposure time has no unit because it is not documented
 */
#define MIN_EXPOSURE_TIME 1
#define MAX_EXPOSURE_TIME 199000
#define DEFAULT_EXPOSURE 30000
#define MIN_GAIN 0
#define MAX_GAIN 244
#define DEFAULT_GAIN 100

#define CANCEL 10

struct camera_vars;

typedef unsigned char uchar;

int camera_device_init(struct camera_vars **device_vars,
		const char *dev_name, int exposure, int gain);
int camera_device_done(struct camera_vars *device_vars);

int camera_device_read_frame(struct camera_vars *device_vars,
		uchar **frame);

unsigned int camera_device_get_width(struct camera_vars *device_vars);
unsigned int camera_device_get_height(struct camera_vars *device_vars);
uint32_t camera_device_get_pixelformat(struct camera_vars *device_vars);

#endif
