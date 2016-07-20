Debayer Project using CUDA
==========================

This Project implements a CUDA application with camera input support over v4l2
library. A CUDA Kernel computes an RGB image from the captured Bayer pattern
images.

* Compile: ./autogen.sh
		make
    * Compilation with OpenCV support.

* Usage: 	cuda_debayer [-h|--help]
* or:		cuda_debayer [-d|--device=/dev/video0] [-e|--exposure=30000]
			[-g|--gain=100]
	* on a platform with OpenCV runtime library

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
        - Outputs stream with OpenCV
