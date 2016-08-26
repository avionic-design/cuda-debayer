noinst_LIBRARIES = lib_camera_device_cuda.a

lib_camera_device_cuda_a_CPPFLAGS = \
	-I$(top_srcdir)/include

lib_camera_device_cuda_a_SOURCES = \
	camera_device.cpp \
	bayer2rgb.cu \
	gl_display.cpp

lib_camera_device_cuda_a_CXXFLAGS = \
	$(OPENGL_CFLAGS) \
	$(CUDA_CFLAGS) \
	$(WARNCFLAGS) \
	$(DEBUGCFLAGS)

AM_NVCC_CXXFLAGS = \
	$(CUDA_CFLAGS) \
	$(addprefix -Xcompiler ,$(WARNCFLAGS) $(DEBUGCFLAGS)) \
	$(DEBUGNVCCFLAGS) \
	$(NVCC_CROSSFLAGS)

.cu.o : ; $(NVCC) \
	$(lib_camera_device_cuda_a_CPPFLAGS) $(CPPFLAGS) \
	$(AM_NVCC_CXXFLAGS) $(NVCC_CXXFLAGS) -c -o $@ $<
