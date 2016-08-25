AC_PREREQ(2.61)
AC_INIT([cuda_debayer], [0.2.1])
AC_CONFIG_SRCDIR([bin/cuda_debayer.cpp])
AC_CONFIG_HEADER(config.h)
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE([no-dist-gzip dist-xz foreign])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE

PKG_PROG_PKG_CONFIG

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

AC_ARG_WITH([opencv],
		[AS_HELP_STRING([--without-opencv],
				[Compile without OpenCV])],
		[], [with_opencv=yes])

AS_IF([test "x$with_opencv" != xno],
	[AC_MSG_NOTICE([with OpenCV])
	PKG_CHECK_MODULES([OPENCV], [opencv >= 2.2],
		AC_DEFINE([HAVE_OPENCV], [1], [Define if you have OpenCV libs]))]
)

AC_ARG_WITH([opengl],
		[AS_HELP_STRING([--without-opengl],
				[Compile without OpenGL])],
		[], [with_opengl=yes])

AS_IF([test "x$with_opengl" != xno], [
	AC_MSG_NOTICE([with OpenGL])
	PKG_CHECK_MODULES([OPENGL], [gl glu glew freeglut])
	AC_DEFINE([HAVE_OPENGL], [1], [Define if you have OpenGL libs])
])

AS_IF([test "x$enable_debug" = "xyes"], [
	AC_DEFINE([ENABLE_DEBUG], [1], [whether debugging is enabled])
	DEBUGCFLAGS="-O0 -g3"
	DEBUGNVCCFLAGS="-G"
])
AC_SUBST([DEBUGCFLAGS])
AC_SUBST([DEBUGNVCCFLAGS])

AC_CHECK_HEADERS([linux/videodev2.h], [], [AC_MSG_ERROR([v4l2 not found])])

AX_PROG_NVCC
PKG_CHECK_MODULES([CUDA], [cudart-6.5])

# Set gencode flags for compilation and linking.
# Gencode flags allow compilation and linking for GPU compute
# capabilities independent of the GPU on the build host.
# The compute capability features are described in
# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
# The compute capability of the GPU can be read with the NVIDIA application
# deviceQuery, which can be found in the SDK.
# cuda-debayer requires minimum compute capability 3.0. Since the
# package has been developed for the target platform TK1, which has
# gpu compute capability 3.2, generate code for that by default.
AS_IF([test "x${NVCC_CXXFLAGS+set}" != xset], [
       NVCC_CXXFLAGS="-gencode arch=compute_32,code=sm_32 -rdc=true"
])

# Unit tests wich check
AS_IF([test "x$cross_compiling" != xyes], [
	PKG_CHECK_MODULES([CHECK], [check >= 0.9.4],
		[have_check=yes], [have_check=no])
])
AM_CONDITIONAL([BUILD_TEST], [test "x$have_check" = xyes])

AS_IF([test "x$USE_NLS" = "xyes"], [
	AC_DEFINE([USE_NLS], [1], [Define to 1 if NLS support is enabled])
])

AC_SUBST([WARNCFLAGS], [-Wall])

AC_ARG_ENABLE([werror],
	[AS_HELP_STRING([--disable-werror],
		[Treat warnings as errors (default: enabled)])],
		[enable_werror="$enableval"],
		[enable_werror=yes])
if test "x$enable_werror" = "xyes"; then
	WARNCFLAGS="$WARNCFLAGS -Werror"
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
	tests/Makefile
	Makefile
])
