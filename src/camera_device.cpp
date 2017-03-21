/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file implements the functionality of initalisation, destruction and
 * running of a v4l2 camera. Private functions provides changing of camera
 * parameters over a shell user interface.
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "camera_device.h"

#define CLEAR(x) memset(&(x), 0, sizeof(x))

struct buffer {
	size_t length;
	void *start;
};

struct camera_vars {
	enum v4l2_memory memory;
	unsigned int n_buffers;
	struct buffer *buffers;

	struct v4l2_format format;

	int fd;
};

static int xioctl(int fh, int request, void *arg)
{
	int ret_val;

	do {
		ret_val = ioctl(fh, request, arg);
	} while (ret_val == -1 && errno == EINTR);

	return ret_val;
}

static int set_exposure(int exposure_time, int fd)
{
	struct v4l2_control control;

	CLEAR(control);
	if ((exposure_time <= MAX_EXPOSURE_TIME) &&
			(exposure_time >= MIN_EXPOSURE_TIME)) {
		control.value = exposure_time;
		printf("OK\n");
	} else {
		control.value = DEFAULT_EXPOSURE;
		printf("Set Exposure time to standard value\n");
	}

	/* change the exposure time */
	control.id = V4L2_CID_EXPOSURE;
	if (xioctl(fd, VIDIOC_S_CTRL, &control) == -1) {
		perror("VIDIOC_S_CTRL set exposure time");
		return -errno;
	}

	return 0;
}

static int set_gain(int new_gain, int fd)
{
	struct v4l2_control control;

	CLEAR(control);
	if ((new_gain <= MAX_GAIN) && (new_gain >= MIN_GAIN)) {
		control.value = new_gain;
		printf("OK\n");
	} else {
		control.value = DEFAULT_GAIN;
		printf("Set Analog Gain to standard value\n");
	}

	/* change the analog gain */
	control.id = V4L2_CID_GAIN;
	if (xioctl(fd, VIDIOC_S_CTRL, &control) == -1) {
		perror("VIDIOC_S_CTRL set analog gain");
		return -errno;
	}

	return 0;
}

static int read_frame(struct camera_vars *cam_vars, uchar **frame)
{
	struct v4l2_buffer buf;

	if (cam_vars == NULL)
		return -EINVAL;

	CLEAR(buf);
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = cam_vars->memory;

	if (xioctl(cam_vars->fd, VIDIOC_DQBUF, &buf) == -1) {
		perror("VIDIOC_DQBUF");
		return -errno;
	}

	*frame = (uchar *)cam_vars->buffers[buf.index].start;

	if (xioctl(cam_vars->fd, VIDIOC_QBUF, &buf) == -1) {
		perror("VIDIOC_QBUF");
		return -errno;
	}

	return 0;
}

int camera_device_read_frame(struct camera_vars *cam_vars,
		uchar **frame)
{
	struct timeval tv;
	int ret_val;
	fd_set fds;
	char n[20];

	if (cam_vars == NULL)
		return -EINVAL;

	FD_ZERO(&fds);
	FD_SET(cam_vars->fd, &fds);
	FD_SET(STDIN_FILENO, &fds);

	tv.tv_sec = 0;
	tv.tv_usec = 30000;

	ret_val = select(cam_vars->fd + 1, &fds, NULL, NULL, &tv);
	if (ret_val == -1) {
		if (errno == EINTR)
			/* signal was caught during select, try again */
			return -EAGAIN;
		perror("select");
		return -errno;
	}

	if (ret_val == 0)
		return -EAGAIN;

	if (FD_ISSET(STDIN_FILENO, &fds)) {

		if (fgets(n, sizeof(n), stdin) == NULL)
			return -EIO;

		switch (n[0]) {
		case 'q':
			/* camera_device.h define */
			return CANCEL;
		case 'e': {
			int exposure = atoi(&n[1]);

			printf("Set exposure time to %i \t", exposure);
			ret_val = set_exposure(exposure, cam_vars->fd);
			if (ret_val != 0)
				return ret_val;
			break;
		}
		case 'g': {
			int gain = atoi(&n[1]);

			printf("Set analog gain to %i \t", gain);
			ret_val = set_gain(gain, cam_vars->fd);
			if (ret_val != 0)
				return ret_val;
			break;
		} default:
			printf("%c is not a valid Parameter\n", n[0]);
			break;
		}
	}

	if (FD_ISSET(cam_vars->fd, &fds)) {
		ret_val = read_frame(cam_vars, frame);
		if (ret_val) {
			printf("read_frame failed: %d", errno);
			return ret_val;
		}

		return 0;
	}

	return -EAGAIN;
}

static int stop_capturing(int fd)
{
	enum v4l2_buf_type type;

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (xioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
		perror("VIDIOC_STREAMOFF");
		return -errno;
	}

	return 0;
}

static int start_capturing(struct camera_vars *cam_vars)
{
	enum v4l2_buf_type type;
	struct v4l2_buffer buf;
	unsigned int i;

	if (cam_vars == NULL)
		return -EINVAL;

	for (i = 0; i < cam_vars->n_buffers; ++i) {
		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = cam_vars->memory;
		buf.index = i;
		if (cam_vars->memory == V4L2_MEMORY_USERPTR) {
			buf.m.userptr =
				(unsigned long)cam_vars->buffers[i].start;
			buf.length = cam_vars->buffers[i].length;
		}

		if (xioctl(cam_vars->fd, VIDIOC_QBUF, &buf) == -1) {
			perror("VIDIOC_QBUF");
			return -errno;
		}
	}

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (xioctl(cam_vars->fd, VIDIOC_STREAMON, &type) == -1) {
		perror("VIDIOC_STREAMON");
		return -errno;
	}

	return 0;
}

static int deinit_device(struct camera_vars *cam_vars)
{
	int ret_val = 0;
	unsigned int i;

	if (cam_vars == NULL)
		return -EINVAL;

	for (i = 0; i < (cam_vars->n_buffers); ++i) {
		if (cam_vars->memory == V4L2_MEMORY_USERPTR)
			free(cam_vars->buffers[i].start);
		else
			ret_val = munmap(cam_vars->buffers[i].start,
					cam_vars->buffers[i].length);
		if (ret_val == -1) {
			fprintf(stderr, "munmap error %d, %s\n", errno,
					strerror(errno));
			ret_val = -errno;
			break;
		}
	}

	free(cam_vars->buffers);

	return ret_val;
}

static int init_buffers(struct camera_vars *cam_vars, size_t size)
{
	struct v4l2_requestbuffers req_buf;
	struct v4l2_buffer buf;
	unsigned int i;
	int err;

	if (cam_vars == NULL || size == 0)
		return -EINVAL;

	CLEAR(req_buf);
	req_buf.count = 4;
	req_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req_buf.memory = V4L2_MEMORY_USERPTR;

	/* First try with user pointer, if that fails fallback on mmap */
	err = xioctl(cam_vars->fd, VIDIOC_REQBUFS, &req_buf);
	if (err < 0 && errno == EINVAL) {
		req_buf.memory = V4L2_MEMORY_MMAP;
		err = xioctl(cam_vars->fd, VIDIOC_REQBUFS, &req_buf);
	}
	if (err < 0) {
		if (errno == EINVAL)
			fprintf(stderr, "device does not support"
				" user pointers or memory mapping\n");
		else
			perror("VIDIOC_REQBUFS");
		return -errno;
	}

	if (req_buf.count < 2) {
		fprintf(stderr, "Insufficient buffer memory on device\n");
		return -errno;
	}

	cam_vars->memory = (enum v4l2_memory)req_buf.memory;
	cam_vars->buffers = (buffer *)calloc(req_buf.count,
			sizeof((cam_vars->buffers)[0]));

	if (!cam_vars->buffers) {
		fprintf(stderr, "Out of memory\n");
		return -errno;
	}

	for (i = 0; i < req_buf.count; i++) {

		if (cam_vars->memory == V4L2_MEMORY_USERPTR) {
			err = posix_memalign(&cam_vars->buffers[i].start,
					sysconf(_SC_PAGESIZE), size);
			if (err) {
				err = -errno;
				fprintf(stderr, "memalig error %d, %s\n",
					errno, strerror(errno));
				goto cleanup;
			}
			cam_vars->buffers[i].length = size;
			cam_vars->n_buffers = i + 1;
			continue;
		}

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (xioctl(cam_vars->fd, VIDIOC_QUERYBUF, &buf) == -1) {
			err = -errno;
			perror("VIDIOC_QUERYBUF");
			goto cleanup;
		}

		cam_vars->buffers[i].length = buf.length;
		cam_vars->buffers[i].start =
				mmap(NULL, buf.length, PROT_READ,
					MAP_SHARED, cam_vars->fd,
					buf.m.offset);

		if (cam_vars->buffers[i].start == MAP_FAILED) {
			err = -errno;
			fprintf(stderr, "mmap error %d, %s\n", errno,
					strerror(errno));
			goto cleanup;
		}

		cam_vars->n_buffers = i + 1;
	}

	return 0;
cleanup:
	for (i = 0; i < cam_vars->n_buffers; i++) {
		if (cam_vars->memory == V4L2_MEMORY_USERPTR)
			free(cam_vars->buffers[i].start);
		else
			munmap(cam_vars->buffers[i].start,
				cam_vars->buffers[i].length);
	}
	free(cam_vars->buffers);

	return err;
}

static int init_device(struct camera_vars *cam_vars,
		const char *dev_name)
{
	struct v4l2_capability cap;
	struct v4l2_format format;
	const char *fmt_name;
	unsigned int size;
	unsigned int min;

	if (cam_vars == NULL)
		return -EINVAL;

	if (xioctl(cam_vars->fd, VIDIOC_QUERYCAP, &cap) == -1) {
		if (errno == EINVAL) {
			fprintf(stderr, "%s is no V4L2 device\n",
					dev_name);
		} else {
			perror("VIDIOC_QUERYCAP");
		}
		return -errno;
	}

	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
		fprintf(stderr, "%s is no video capture device\n",
				dev_name);
		return -errno;
	}

	if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
		fprintf(stderr, "%s does not support streaming i/o\n",
				dev_name);
		return -errno;
	}

	CLEAR(format);
	format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (xioctl(cam_vars->fd, VIDIOC_G_FMT, &format) == -1) {
		perror("VIDIOC_G_FMT");
		return -errno;
	}

	switch (format.fmt.pix.pixelformat) {
	case V4L2_PIX_FMT_SBGGR8:
		fmt_name = "BGGR";
		break;
	case V4L2_PIX_FMT_SGBRG8:
		fmt_name = "GBRG";
		break;
	case V4L2_PIX_FMT_SGRBG8:
		fmt_name = "GRBG";
		break;
	case V4L2_PIX_FMT_SRGGB8:
		fmt_name = "RGGB";
		break;
	case V4L2_PIX_FMT_SBGIR8:
		fmt_name = "BGIR";
		break;
	case V4L2_PIX_FMT_SGBRI8:
		fmt_name = "GBRI";
		break;
	case V4L2_PIX_FMT_SIRBG8:
		fmt_name = "IRBG";
		break;
	case V4L2_PIX_FMT_SRIGB8:
		fmt_name = "RIGB";
		break;
	default:
		fmt_name = "unknown";
		break;
	}

	printf("Format %s %ix%i (stride = %i bytes)\n", fmt_name,
		format.fmt.pix.width, format.fmt.pix.height,
		format.fmt.pix.bytesperline);

	min = format.fmt.pix.width * 2;

	if (format.fmt.pix.bytesperline < min)
		format.fmt.pix.bytesperline = min;

	size = format.fmt.pix.bytesperline * format.fmt.pix.height;

	if (format.fmt.pix.sizeimage < size)
		format.fmt.pix.sizeimage = size;

	if (init_buffers(cam_vars, size) != 0)
		return -errno;

	cam_vars->format = format;

	return 0;
}

static int close_device(int fd)
{
	if (fd < 0) {
		fprintf(stderr, "file descriptor not available\n");
		return -EBADFD;
	}

	if (close(fd) == -1) {
		perror("Error at close fd");
		return -errno;
	}

	return 0;
}

static int open_device(const char *dev_name, int *fd)
{
	struct stat status;

	if (stat(dev_name, &status) == -1) {
		fprintf(stderr, "Cannot identify '%s': %d, %s\n",
				dev_name, errno, strerror(errno));
		return -errno;
	}

	if (!S_ISCHR(status.st_mode)) {
		fprintf(stderr, "%s is no device\n", dev_name);
		return -errno;
	}

	*fd = open(dev_name, O_RDWR | O_NONBLOCK, 0);

	if (*fd == -1) {
		fprintf(stderr, "Cannot open '%s': %d, %s\n",
				dev_name, errno, strerror(errno));
		return -errno;
	}

	return 0;
}

unsigned int camera_device_get_width(struct camera_vars *cam_vars)
{
	if (cam_vars == NULL)
		return 0;

	return cam_vars->format.fmt.pix.width;
}

unsigned int camera_device_get_height(struct camera_vars *cam_vars)
{
	if (cam_vars == NULL)
		return 0;

	return cam_vars->format.fmt.pix.height;
}

uint32_t camera_device_get_pixelformat(struct camera_vars *cam_vars)
{
	if (cam_vars == NULL)
		return 0;

	return cam_vars->format.fmt.pix.pixelformat;
}

int camera_device_init(struct camera_vars **cam_vars_p,
		const char  *dev_name, int exposure, int gain)
{
	struct camera_vars *cam_vars;
	int ret_val = 0;

	if (cam_vars_p == NULL)
		return -EINVAL;

	cam_vars = (camera_vars *) malloc(sizeof(struct camera_vars));
	if (!cam_vars)
		return -ENOMEM;

	ret_val = open_device(dev_name, &cam_vars->fd);
	if (ret_val)
		return ret_val;

	ret_val = init_device(cam_vars, dev_name);
	if (ret_val)
		return ret_val;

	if (exposure > 0)
		ret_val = set_exposure(exposure, cam_vars->fd);

	if (gain > 0)
		ret_val = set_gain(gain, cam_vars->fd);

	ret_val = start_capturing(cam_vars);
	if (ret_val)
		return -ret_val;

	*cam_vars_p = cam_vars;

	return 0;
}

int camera_device_done(struct camera_vars *cam_vars)
{
	int ret_val = 0;

	if (cam_vars == NULL)
		return -EINVAL;

	ret_val = stop_capturing(cam_vars->fd);
	if (ret_val != 0)
		goto cleanup;

	ret_val = deinit_device(cam_vars);
	if (ret_val != 0)
		goto cleanup;

	ret_val = close_device(cam_vars->fd);
	if (ret_val != 0)
		goto cleanup;

	ret_val = EXIT_SUCCESS;

cleanup:
	free(cam_vars);

	return ret_val;
}
