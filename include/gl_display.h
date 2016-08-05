/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#ifndef GL_DISPLAY_H
#define GL_DISPLAY_H

struct gl_display_vars;

int gl_display_init(gl_display_vars **gl_vars_p, uint32_t width,
		uint32_t height, float scale, int argc, char **argv);
int gl_display_free(gl_display_vars *gl_vars);

int gl_display_show(gl_display_vars *gl_vars, uint32_t *odata,
		cudaStream_t stream);
int gl_display_fps(gl_display_vars *gl_vars, double fps);

#endif
