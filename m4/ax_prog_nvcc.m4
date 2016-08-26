dnl  @synopsis AX_PROG_NVCC
dnl  @summary Set NVCC, declare NVCC_CXXFLAGS and provide NVCC_CROSSFLAGS
dnl
dnl  AX_PROG_NVCC sets up the two precious variables:
dnl        NVCC           - The nvcc cuda compiler driver
dnl        NVCC_CXXFLAGS  - nvcc specific compiler flags
dnl
dnl  Plus, it defines the convenience variable
dnl        NVCC_CROSSFLAGS - auto-detected nvcc cross-compiling flags
dnl
dnl  It also calls AC_PROG_CXX to determine the (host) c++ compiler.
dnl  nvcc is NOT set as the standard host c++ compiler, CXX is not
dnl  changed.
dnl
dnl  NVCC is set to nvcc, if found.
dnl
dnl  When cross-compiling, NVCC_CROSSFLAGS is defined to setup nvcc
dnl  accordingly. This includes setting -ccbin to ${CXX}.
dnl  Currently only ARM is supported with NVCC_CROSSFLAGS.
dnl
dnl  Setting up the necessary _CFLAGS and _LIBS is left to the user.
dnl
dnl  @version 2016-08-26
dnl  @author Nikolaus Schulz <nikolaus.schulz@avionic-design.de>
dnl  @license GNU All-Permissive

AC_DEFUN([AX_PROG_NVCC], [

AC_PROG_CXX

AC_ARG_VAR([NVCC_CXXFLAGS], [nvcc CUDA compiler flags])

AS_IF([test "x$cross_compiling" = xyes],[
	NVCC_CROSSFLAGS="-ccbin ${CXX}"
	case "${host_cpu}" in
	arm*)
		NVCC_CROSSFLAGS="${NVCC_CROSSFLAGS} -m32 -target-cpu-arch ARM" ;;
	aarch64*)
		NVCC_CROSSFLAGS="${NVCC_CROSSFLAGS} -m64 -target-cpu-arch ARM" ;;
	esac
])
AC_SUBST([NVCC_CROSSFLAGS])

AC_ARG_VAR([NVCC], [the CUDA compiler driver])
AC_PATH_PROG([NVCC], [nvcc], [missing])
AS_IF([test "x${NVCC}" = xmissing],[
	AC_MSG_ERROR([cannot find nvcc])
])

])
