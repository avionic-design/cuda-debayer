AC_PREREQ(2.61)
AC_INIT([cuda_debayer], [0.1])
AC_CONFIG_SRCDIR([bin/cuda_debayer.cpp])
AC_CONFIG_HEADER(config.h)
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE([no-dist-gzip dist-xz foreign])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE

AM_PROG_CC_C_O
AC_PROG_RANLIB
AC_PROG_INSTALL
AM_NLS

AC_MSG_CHECKING([whether to enable debugging])
AC_ARG_ENABLE([debug],
	[AS_HELP_STRING([--enable-debug],
			[Enable debugging [default=no]])],
	[], [enable_debug=no])
AC_MSG_RESULT([$enable_debug])

AC_ARG_WITH([opencv], [  --without-opencv        Compile without OpenCV],
		[], [with_opencv=yes])

AS_IF([test "x$with_opencv" != xno],
	[AC_MSG_NOTICE([with OpenCV])
	AC_MSG_CHECKING([for OpenCV 2.2.x or newer])
	AC_CHECK_LIB(opencv_core, cvCreateImage, [opencv="yes";
		AC_DEFINE(HAVE_OPENCV_22X, 1,
		[Define HAVE_OPENCV_22X if OpenCV 2.2.x is installed])])
	AC_DEFINE([HAVE_OPENCV], [1], [Define if you have OpenCV libs])

	AS_IF([test "x$opencv" = "xyes"],
		[OPENCV_LD_FLAGS=`pkg-config --libs opencv`
		AC_SUBST(OPENCV_LD_FLAGS)],
		[AC_MSG_ERROR([OpenCV 2.2 or greater not found])])], []
)

AC_ARG_WITH([opengl], [  --without-opengl        Compile without OpenGL],
		[], [with_opengl=yes])

AS_IF([test "x$with_opengl" != xno],
	[AC_MSG_NOTICE([with OpenGL])
	AC_CHECK_LIB(GL, glBegin, [GL_LIBS=-lGL], AC_MSG_ERROR([GL required]))
	AC_CHECK_LIB(GLU, gluBeginCurve, [GLU_LIBS=-lGLU], AC_MSG_ERROR([GLU required]))
	AC_CHECK_LIB(GLEW, glewInit, [GLEW_LIBS=-lGLEW], AC_MSG_ERROR([GLEW required]))
	AC_CHECK_LIB(glut, glutInit, [GLUT_LIBS=-lglut], AC_MSG_ERROR([glut required]))
	AC_DEFINE([HAVE_OPENGL], [1], [Define if you have OpenGL libs])

	# Set LD_FLAGS for OpenGL linking
	AC_SUBST(OPENGL_LD_FLAGS, "$GL_LIBS $GLU_LIBS $GLEW_LIBS $GLUT_LIBS")
	AC_SUBST(OPENGL_FLAGS, "$GLEW_CFLAGS $GL_CFLAGS")], []
)

AS_IF([test "x$enable_debug" = "xyes"], [
	AC_DEFINE([ENABLE_DEBUG], [1], [whether debugging is enabled])
	CFLAGS="$CFLAGS -O0 -g3"
])

AC_CHECK_HEADERS([linux/videodev2.h], [], [AC_MSG_ERROR([v4l2 not found])])

AX_CHECK_CUDA

# Set gencode flags for compilation and linking.
# Gencode flags allow compilation and linking for other compute capabilities
# than the highest compute capabilty of the used GPU.
# The compute capability features are discribed in
# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
# The compute capability of the used GPU can be read with the NVIDIA application
# deviceQuery, which can be found in the SDK.
GENCODE_FLAGS="-gencode arch=compute_32,code=sm_32"
AC_SUBST(GENCODE_FLAGS)

# Set CXX compiler to nvcc for linking with CUDA object file
AC_PROG_CXX([nvcc])

AS_IF([test "x$USE_NLS" = "xyes"], [
	AC_DEFINE([USE_NLS], [1], [Define to 1 if NLS support is enabled])
])

# enable all warnings
CFLAGS="$CFLAGS -Wall"

AC_ARG_ENABLE([werror],
	[AS_HELP_STRING([--disable-werror],
		[Treat warnings as errors (default: enabled)])],
		[enable_werror="$enableval"],
		[enable_werror=yes])
if test "x$enable_werror" = "xyes"; then
	CFLAGS="$CFLAGS -Werror"
fi

# Loose the directory print messages in silent mode
if test x"$AM_DEFAULT_VERBOSITY" = x"0"; then
	if make --help 2>&1 | grep -q no-print-directory; then
		AM_MAKEFLAGS="$AM_MAKEFLAGS --no-print-directory"
	fi
fi
AC_SUBST(AM_MAKEFLAGS)

AC_OUTPUT([
	src/Makefile
	bin/Makefile
	Makefile
])
