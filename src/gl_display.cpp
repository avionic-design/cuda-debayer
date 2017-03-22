/*
 * Copyright (C) 2016 Avionic Design GmbH
 * Meike Vocke <meike.vocke@avionic-design.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#include <string>
#include <stdio.h>
#include <errno.h>
#include <X11/Xlib.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "gl_display.h"

/* #define DEBUG */

struct gl_display_vars {
	int width;
	int height;
	GLuint textureID;

	struct cudaGraphicsResource *cuda_pbo_resource;
	int GLUT_window_handle;
	GLuint shader_draw_tex;

	cudaArray *texture_ptr;
};

static void check_gl_error(uint32_t line, std::string string)
{
#ifdef DEBUG
	GLenum err;

	while ((err = glGetError()) != GL_NO_ERROR) {
		printf("%s, %i: %d\t%s\n", string.c_str(), line, err,
				gluErrorString(err));
	}
#endif
}

static void check_program_info_log(GLuint program)
{
#ifdef DEBUG
	int infologLength = 0;
	int charsWritten  = 0;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

	if (infologLength > 1) {
		char *infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(program, infologLength, (GLsizei *)&charsWritten,
				infoLog);
		printf("Shader compilation error: %s\n", infoLog);
		free(infoLog);
	}
#endif
}

static bool compile_shader(GLuint shader, const char *shader_src)
{
	char temp[256] = "";

	glShaderSource(shader, 1, &shader_src, NULL);
	glCompileShader(shader);

	/* check if shader compiled */
	GLint compiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

	if (compiled)
		return true;

	glGetShaderInfoLog(shader, 256, NULL, temp);
	printf("shader compilation failed:\n%s\n", temp);
	glDeleteShader(shader);

	return false;
}

static int compile_GLSL_program(GLuint *program_p)
{
	GLuint v, f = 0;
	GLuint program;

	/*
	 * The GLSL shader code has to be defined as a char pointer,
	 * otherwise a external shader file has to be loaded during runtime.
	 * Therefore the shader files had to exist in the same folder as the
	 * binary. IMO this solutions is better than an extra shader file.
	 */
	const char *glsl_drawtex_vertexshader_src =
			"void main(void)\n"
			"{\n"
			"	gl_Position = gl_Vertex;\n"
			"	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
			"}\n";

	const char *glsl_drawtex_fragshader_src =
			"#version 130\n"
			"uniform usampler2D texImage;\n"
			"void main()\n"
			"{\n"
			"	vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
			"	gl_FragColor = c / 255.0;\n"
			"}\n";

	program = glCreateProgram();

	if (glsl_drawtex_vertexshader_src) {
		v = glCreateShader(GL_VERTEX_SHADER);

		if (!compile_shader(v, glsl_drawtex_vertexshader_src))
			return -1;

		glAttachShader(program, v);
	}

	if (glsl_drawtex_fragshader_src) {
		f = glCreateShader(GL_FRAGMENT_SHADER);

		if (!compile_shader(f, glsl_drawtex_fragshader_src))
			return -1;

		glAttachShader(program, f);
	}

	if (!glsl_drawtex_vertexshader_src || !glsl_drawtex_vertexshader_src) {
		printf("missing vertex and/or fragment shader\n");
		return -1;
	}

	glLinkProgram(program);

	*program_p = program;

	check_program_info_log(program);

	return 0;
}

static void reshape_window(int win_width, int win_height)
{
	gl_display_vars *gl_vars = (gl_display_vars *)glutGetWindowData();
	float ratio = (float)gl_vars->width / (float)gl_vars->height;
	int x, y, width, height;

	/* Compute the new size respecting the original aspect ratio */
	width = win_width;
	height = win_width / ratio;
	if (height > win_height) {
		width = win_height * ratio;
		height = win_height;
	}

	/* Center the view in the window */
	x = (width < win_width) ? (win_width - width) / 2 : 0;
	y = (height < win_height) ? (win_height - height) / 2 : 0;

	/* Apply the new view */
	glViewport(x, y, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)width / (GLfloat)height, 0.1f, 10.0f);
	glMatrixMode(GL_MODELVIEW);
}

static void display_window(void)
{
	/* Nothing to do here, but glut require this callback. */
}

static bool initGL(gl_display_vars *gl_vars, float scale, int *argc,
		char **argv)
{
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f }; /* RGBA color for white */
	float red[] = { 1.0f, 0.1f, 0.1f, 1.0f }; /* RGBA color for red */

	/* Create GL context */
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(gl_vars->width * scale, gl_vars->height * scale);
	gl_vars->GLUT_window_handle = glutCreateWindow("Debayer Frames");
	glutDisplayFunc(display_window);
	glutReshapeFunc(reshape_window);
	glutSetWindowData(gl_vars);

	check_gl_error(__LINE__, "error during create GL context");

	/* initialize necessary OpenGL extensions */
	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object "
			"GL_EXT_framebuffer_object ")) {
		printf("ERROR: Support for OpenGL extensions "
				"ARB_pixel_buffer_object, "
				"EXT_framebuffer_object and OpenGL Version 2.0 "
				"missing.");
		fflush(stderr);
		return false;
	}

	check_gl_error(__LINE__, "glew init error");

	/* Specify the RGBA color when color buffers are cleared */
	glClearColor(0.5, 0.5, 0.5, 1.0);

	glDisable(GL_DEPTH_TEST);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	check_gl_error(__LINE__, "error during setting of GL variables");

	glEnable(GL_LIGHT0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

	check_gl_error(__LINE__, "error during specification of front and back "
			"parameters");

	return true;
}

static int createTextureDst(gl_display_vars *gl_vars)
{
	cudaError_t ret_cuda;

	glGenTextures(1, &(gl_vars->textureID));
	glBindTexture(GL_TEXTURE_2D, gl_vars->textureID);

	/* set basic parameters */
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, gl_vars->width,
			gl_vars->height, 0, GL_RGBA_INTEGER_EXT,
			GL_UNSIGNED_BYTE, NULL);

	check_gl_error(__LINE__, "error during create texture");

	/* register this texture with CUDA */
	ret_cuda = cudaGraphicsGLRegisterImage(&(gl_vars->cuda_pbo_resource),
			gl_vars->textureID, GL_TEXTURE_2D,
			cudaGraphicsMapFlagsWriteDiscard);
	if (ret_cuda != cudaSuccess) {
		printf("cudaGraphicsGLRegisterImage\t %s\n",
				cudaGetErrorString(ret_cuda));
		return -EINVAL;
	}

	return 0;
}

int gl_display_init(gl_display_vars **gl_vars_p, uint32_t width,
		uint32_t height, float scale, int argc, char **argv)
{
	gl_display_vars *gl_vars;
	cudaError_t ret_cuda;
	int ret_val;

	gl_vars = (gl_display_vars *)malloc(sizeof(struct gl_display_vars));

	if (gl_vars == NULL) {
		printf("Error in malloc");
		ret_val = -ENOMEM;
		goto cleanup;
	}

	gl_vars->width = width;
	gl_vars->height = height;

	if (initGL(gl_vars, scale, &argc, argv) == false) {
		printf("failed to init GL\n");
		ret_val = -ENOSYS;
		goto cleanup;
	}

	cudaGLSetGLDevice(0);

	ret_val = createTextureDst(gl_vars);
	if (ret_val != 0) {
		printf("failed to create GL texture");
		goto cleanup;
	}

	ret_val = compile_GLSL_program(&(gl_vars->shader_draw_tex));
	if (ret_val)
		goto cleanup;

	check_gl_error(__LINE__, "compile shader error");

	ret_cuda = cudaGraphicsMapResources(1, &(gl_vars->cuda_pbo_resource), 0);
	if (ret_cuda != cudaSuccess) {
		printf("cudaGraphicsMapResources\t %s\n",
				cudaGetErrorString(ret_cuda));
		ret_val = -EINVAL;
		goto cleanup;
	}

	ret_cuda = cudaGraphicsSubResourceGetMappedArray(&(gl_vars->texture_ptr),
			gl_vars->cuda_pbo_resource, 0, 0);
	if (ret_cuda != cudaSuccess) {
		printf("cudaGraphicsSubResourceGetMappedArray\t %s\n",
				cudaGetErrorString(ret_cuda));
		ret_val = -EINVAL;
		goto cleanup;
	}

	cudaGraphicsUnmapResources(1, &(gl_vars->cuda_pbo_resource), 0);

	*gl_vars_p = gl_vars;

	return 0;

cleanup:
	gl_display_free(gl_vars);
	return ret_val;
}

int gl_display_free(gl_display_vars *gl_vars)
{
	if (gl_vars == NULL)
		return -EINVAL;

	cudaGraphicsUnregisterResource(gl_vars->cuda_pbo_resource);

	glDeleteTextures(1, &(gl_vars->textureID));

	cudaDeviceReset();

	if (gl_vars->GLUT_window_handle)
		glutDestroyWindow(gl_vars->GLUT_window_handle);

	free(gl_vars);

	return 0;
}

static int displayImage(GLuint texture, GLuint shader_draw_tex, unsigned int width,
		unsigned int height)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	check_gl_error(__LINE__, "prepare texture for displaying error");

	glUseProgram(shader_draw_tex);
	GLint id = glGetUniformLocation(shader_draw_tex, "texImage");
	glUniform1i(id, 0);

	check_gl_error(__LINE__, "use loaded shader for texture error");

	/*
	 * specify which points in the texture are referenced to the points in
	 * the rectangle of the GL_QUADS
	 */
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(1.0, 1.0, 0.5);
	glEnd();

	check_gl_error(__LINE__, "draw GL_QUADS with texture error");

	glDisable(GL_TEXTURE_2D);

	glUseProgram(0);

	check_gl_error(__LINE__, "switch to default shader error");

	return 0;
}

int gl_display_fps(gl_display_vars *gl_vars, double fps)
{
	char tmp_string[50];

	sprintf(tmp_string, "%f fps   Debayer Frames", fps);

	if (fps > 0.1f)
		glutSetWindowTitle(tmp_string);

	return 0;
}

int gl_display_show(gl_display_vars *gl_vars, uint32_t *odata,
		cudaStream_t stream)
{
	int size_tex_data = gl_vars->width * gl_vars->height *
			sizeof(uint32_t);
	cudaError_t ret_cuda;

	if (gl_vars == NULL) {
		printf("invalid gl_vars parameter\n");
		return -EINVAL;
	}

	ret_cuda = cudaMemcpyToArrayAsync(gl_vars->texture_ptr, 0, 0, odata,
			size_tex_data, cudaMemcpyDefault, stream);
	if (ret_cuda != cudaSuccess) {
		printf("MemcpyToArray, %s\n",
				cudaGetErrorString(ret_cuda));
		return -EINVAL;
	}

	cudaStreamSynchronize(stream);

	displayImage(gl_vars->textureID, gl_vars->shader_draw_tex,
			gl_vars->width, gl_vars->height);

	/*
	 * draw on backbuffer and flip it with frontbuffer after all drawings
	 * are done
	 */
	glutSwapBuffers();

	/* Let glut run its event handlers */
	glutMainLoopEvent();

	return 0;
}
