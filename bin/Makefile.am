cuda_debayer_LDADD = \
	$(top_build_prefix)/src/lib_camera_device_cuda.a \
	$(OPENCV_LIBS) \
	$(CUDA_LIBS) \
	$(OPENGL_LIBS)

bin_PROGRAMS = \
	cuda_debayer

cuda_debayer_CPPFLAGS = \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/include

cuda_debayer_CXXFLAGS = \
	$(OPENCV_CFLAGS) \
	$(OPENGL_CFLAGS) \
	$(CUDA_CFLAGS) \
	$(WARNCFLAGS) \
	$(DEBUGCFLAGS)

cuda_debayer_SOURCES = \
	cuda_debayer.cpp

AM_NVCC_CXXFLAGS = \
	$(CUDA_CFLAGS) \
	$(addprefix -Xcompiler ,$(WARNCFLAGS) $(DEBUGCFLAGS)) \
	$(DEBUGNVCCFLAGS)

cuda_debayer_LINK = $(NVCC) \
	$(AM_NVCC_CXXFLAGS) $(NVCC_CXXFLAGS) \
	-o $@
