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

#include <string.h>

/*
 * exposure time has no unit because it is not documented
 */
#define MIN_EXPOSURE_TIME 1
#define MAX_EXPOSURE_TIME 199000
#define DEFAULT_EXPOSURE 100000
#define MIN_GAIN 0
#define MAX_GAIN 244
#define DEFAULT_GAIN 100

#define CANCEL 10

struct camera_vars;

typedef unsigned char uchar;

int camera_device_init(struct camera_vars **device_vars,
		std::string dev_name, int exposure, int gain);
int camera_device_done(struct camera_vars *device_vars);

int camera_device_read_frame(struct camera_vars *device_vars,
		uchar **frame);

unsigned int camera_device_get_width(struct camera_vars *device_vars);
unsigned int camera_device_get_height(struct camera_vars *device_vars);

#endif
