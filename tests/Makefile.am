TESTS = check_cuda_kernel

check_PROGRAMS = check_cuda_kernel

check_cuda_kernel_SOURCES = \
	check_cuda_kernel.cu

check_cuda_kernel_CPPFLAGS = \
	-I$(top_srcdir)/src

check_cuda_kernel_CXXFLAGS = \
	$(CHECK_CFLAGS) \
	$(CUDA_CFLAGS) \
	$(WARNCFLAGS) \
	$(DEBUGCFLAGS)

check_cuda_kernel_LDADD_orig = \
	$(top_build_prefix)src/lib_camera_device_cuda.a \
	$(CHECK_LIBS) \
	$(CUDA_LIBS)

check_cuda_kernel_LINK = $(NVCC) \
	$(AM_NVCC_CXXFLAGS) $(NVCC_CXXFLAGS) \
	-o $@

# nvcc chokes on -pthread
NVCC_STRIP_OPTS = -pthread
check_cuda_kernel_LDADD = \
	$(filter-out $(NVCC_STRIP_OPTS),$(check_cuda_kernel_LDADD_orig))

# When automake computes _DEPENDENCIES, it can't handle the dynamic
# check_cuda_kernel_LDADD, therefore specify _DEPENDENCIES explicitly.
check_cuda_kernel_DEPENDENCIES =

AM_NVCC_CXXFLAGS = \
	$(filter-out $(NVCC_STRIP_OPTS),$(CUDA_CFLAGS) $(CHECK_CFLAGS)) \
	$(addprefix -Xcompiler ,$(WARNCFLAGS) $(DEBUGCFLAGS)) \
	$(DEBUGNVCCFLAGS) \
	$(NVCC_CROSSFLAGS)

.cu.o : ; $(NVCC) \
	$(check_cuda_kernel_CPPFLAGS) $(CPPFLAGS) \
	$(AM_NVCC_CXXFLAGS) $(NVCC_CXXFLAGS) -c -o $@ $<
