Debayer Project using CUDA
==========================

This Project implements a CUDA application with camera input support over v4l2
library. A CUDA Kernel computes an RGB image from the captured Bayer pattern
images. The RGB image stream can be displayed with OpenCV or OpenGL. At
configure time the support of OpenGL or OpenCV can be disabled. Both sinks are
enabled by default.

* Compile: ./autogen.sh [--without-{opencv|opengl}]
		make
    * Compilation with OpenCV and OpenGL support.

* Usage: 	cuda_debayer [-h|--help]
* or:		cuda_debayer [-d|--device=/dev/video0] [-e|--exposure=30000]
			[-g|--gain=100] [-o|--output=opengl] [-s|--scale=1]
	* on a platform with OpenCV and OpenGL runtime library

+    Options:
    - -d | --device
        - Video device name [default:/dev/video0]
    - -e | --exposure
        - Set exposure time   [1..199000; default: 30000]
    - -g | --gain
        - Set analog gain [0..244; default: 100]
    - -h | --help
        - Print help message
    - -o | --output
        - Outputs stream over OpenCV/OpenGL [opencv, opengl; default: opengl]
    - -s | --scale
        - Decreasing size by factor [1..20; default: 1]
