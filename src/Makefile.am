noinst_LIBRARIES = lib_camera_device_cuda.a

lib_camera_device_cuda_a_CPPFLAGS = \
	-I$(top_srcdir)/include

lib_camera_device_cuda_a_SOURCES = \
	camera_device.cpp \
	bayer2rgb.cu \
	gl_display.cpp

lib_camera_device_cuda_a_CXXFLAGS = \
	$(OPENGL_CFLAGS)

NVCC_CPPFLAGS = $(CUDA_CFLAGS) \
	-I$(top_srcdir)/include

NVCC_CXXFLAGS = -O2 -g -G -dc \
	$(GENCODE_FLAGS)

.cu.o : ; $(NVCC) $(NVCC_CPPFLAGS) $(NVCC_CXXFLAGS) -c -o $@ $<
