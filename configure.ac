AC_PREREQ(2.61)
AC_INIT([cuda_debayer], [0.1])
AC_CONFIG_SRCDIR([bin/cuda_debayer.cpp])
AC_CONFIG_HEADER(config.h)
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE([no-dist-gzip dist-xz foreign])
AM_MAINTAINER_MODE

AC_PROG_CXX
AM_PROG_CC_C_O
AC_PROG_INSTALL
AM_NLS

AC_MSG_CHECKING([whether to enable debugging])
AC_ARG_ENABLE([debug],
	[AS_HELP_STRING([--enable-debug],
			[Enable debugging [default=no]])],
	[], [enable_debug=no])
AC_MSG_RESULT([$enable_debug])

AS_IF([test "x$enable_debug" = "xyes"], [
	AC_DEFINE([ENABLE_DEBUG], [1], [whether debugging is enabled])
	CFLAGS="$CFLAGS -O0 -g3"
])

AC_CHECK_HEADERS([readline/readline.h])
AC_CHECK_LIB([readline], [readline],
	[lv_use_readline=yes],
	[lv_use_readline=no])

AC_MSG_CHECKING([for OpenCV 2.2.x or newer])
AC_CHECK_LIB(opencv_core, cvCreateImage, [opencv="yes";
	AC_DEFINE(HAVE_OPENCV_22X, 1,
	[Define HAVE_OPENCV if OpenCV installed])])

AS_IF([test "x$opencv" = "xyes"],
	[OPENCV_LD_FLAGS=`pkg-config --libs opencv`
	AC_SUBST(OPENCV_LD_FLAGS)],
	[AC_MSG_ERROR([OpenCV 2.2 or greater not found])])

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

AC_OUTPUT([
	bin/Makefile
	Makefile
])
