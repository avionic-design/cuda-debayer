TESTS = check_cuda_kernel

check_PROGRAMS = check_cuda_kernel

check_cuda_kernel_SOURCES = \
	check_cuda_kernel.cu

check_cuda_kernel_CPPFLAGS = \
	-I$(top_srcdir)/src \
	$(CUDA_CFLAGS) \
	$(GENCODE_FLAGS)

check_cuda_kernel_LDADD_orig = \
	$(top_build_prefix)src/lib_camera_device_cuda.a \
	$(CHECK_LIBS) \
	$(CUDA_LDFLAGS) \
	$(GENCODE_FLAGS)

check_cuda_kernel_LINK = $(NVCC) -o $@

# nvcc chokes on -pthread
NVCC_STRIP_OPTS = -pthread
check_cuda_kernel_LDADD = \
	$(filter-out $(NVCC_STRIP_OPTS),$(check_cuda_kernel_LDADD_orig))

# When automake computes _DEPENDENCIES, it can't handle the dynamic
# check_cuda_kernel_LDADD, therefore specify _DEPENDENCIES explicitly.
check_cuda_kernel_DEPENDENCIES =

NVCC_CPPFLAGS = $(CUDA_CFLAGS) \
	-I$(top_srcdir)/src

NVCC_CXXFLAGS = -O2 -g -G -dc \
	$(GENCODE_FLAGS)

.cu.o : ; $(NVCC) $(NVCC_CPPFLAGS) $(NVCC_CXXFLAGS) -c -o $@ $<
