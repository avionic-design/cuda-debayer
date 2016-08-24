cuda_debayer_LDADD = \
	$(top_build_prefix)/src/lib_camera_device_cuda.a \
	$(OPENCV_LIBS) \
	$(CUDA_LDFLAGS) \
	$(GENCODE_FLAGS) \
	$(OPENGL_LIBS)

bin_PROGRAMS = \
	cuda_debayer

cuda_debayer_CPPFLAGS = \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/include

cuda_debayer_CXXFLAGS = \
	$(OPENCV_CFLAGS) \
	$(OPENGL_CFLAGS) \
	$(CUDA_CFLAGS)

cuda_debayer_SOURCES = \
	cuda_debayer.cpp
