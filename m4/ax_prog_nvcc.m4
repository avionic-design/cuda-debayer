dnl  @synopsis AX_PROG_NVCC
dnl  @summary Set NVCC and initialize NVCC_CXXFLAGS
dnl
dnl  AX_PROG_NVCC sets up the two precious variables:
dnl        NVCC           - The nvcc cuda compiler driver
dnl        NVCC_CXXFLAGS  - nvcc specific compiler flags
dnl
dnl  It also calls AC_PROG_CXX to determine the (host) c++ compiler.
dnl  nvcc is NOT set as the standard host c++ compiler.
dnl
dnl  NVCC_CXXFLAGS is defined to include the -ccbin option pointing to
dnl  the c++ compiler determined by CXX, and if cross-compiling, the
dnl  target cpu architecture.
dnl
dnl  Setting up the necessary _CFLAGS and _LIBS is left to the user.
dnl
dnl  @version 2016-08-22
dnl  @author Nikolaus Schulz <nikolaus.schulz@avionic-design.de>
dnl  @license GNU All-Permissive

AC_DEFUN([AX_PROG_NVCC], [

AC_PROG_CXX

AC_ARG_VAR([NVCC_CXXFLAGS], [nvcc CUDA compiler flags])
NVCC_CXXFLAGS="-ccbin ${CXX} ${NVCC_CXXFLAGS}"
AS_IF([test "x$cross_compiling" = xyes],[
	case "${host_cpu}" in
	arm*)
		NVCC_CXXFLAGS="-m32 -target-cpu-arch ARM ${NVCC_CXXFLAGS}" ;;
	aarch64*)
		NVCC_CXXFLAGS="-m64 -target-cpu-arch ARM ${NVCC_CXXFLAGS}" ;;
	esac
])

AC_ARG_VAR([NVCC], [the CUDA compiler driver])
AC_PATH_PROG([NVCC], [nvcc], [missing])
AS_IF([test "x${NVCC}" = xmissing],[
	AC_MSG_ERROR([cannot find nvcc])
])

])
