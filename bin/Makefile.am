cuda_debayer_LDADD = \
	$(top_build_prefix)/src/lib_camera_device.a \
	@OPENCV_LD_FLAGS@

bin_PROGRAMS = \
	cuda_debayer

cuda_debayer_CPPFLAGS = \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/include

cuda_debayer_SOURCES = \
	cuda_debayer.cpp
