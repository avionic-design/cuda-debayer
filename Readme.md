Debayer Project for CUDA
========================

This Project implements a Cuda application with camera input support over v4l2
library.

* Compile: ./autogen.sh
		make
    * Compilation with OpenCV support.

* Usage: ./bayer2rgb [options] on a platform with OpenCV runtime library

    * Options:
    * -h   --help          Print this message
    * -o   --output        Outputs stream to stdout
