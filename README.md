# Parallel-Unsharp-Mask
Authored by Jack Smith

## Information:
Created using Visual Studio using C++ and OpenCL.

### Dependencies:
This project comes with the dependencies prepackaged. All you need to do to build the project is to run a CMake cycle:
`cmake -S . -B build`

## Purpose:
Unsharp mask that is parallelised using OpenCL. Loads an image from file, processes it and then writes the result to another file to be viewed.

## Explanation:
An Unsharp Mask is an image sharpening technique that predates image processing on computers. Its purpose is to improve the apparent resolution of an image, it does not actually give the image a larger resolution but manipulates it to make it appear as such. This is achieved by subtracting a blurred version of the image from the original. The result is that the image appears to have sharper edges and it looks more defined because the noise of the blurry image has been removed.

In order to parallelise the unsharp mask the approach used was to parallelise the individual serial steps. This made sense because it would keep the code close to the logic of the original and allow a user to see the similarity between the two.

See the docs for a full explanation on how serial code was parrlellised using OpenCL and bench test comparison of Serial vs Parallel.

## TLDR:
In conclusion, the serial program was adapted to function in a naïve parallel manner using global memory that was processed in parallel in multiple threads. This made it more effective than the serial version by up to a factor of 130. The more the blur size increased, the larger the gains the parallelised system had over the serial one. The GPU performed best on parallelism against the CPU in this case, it was able to complete faster than the CPU in all cases by up to a factor of 16. This advantage came through its larger thread capacity which meant more pixels could be processed at once. With careful use of the OpenCL API this program has been accelerated and the amount of acceleration will only improve as the problem size expands. As time passes and multithreaded hardware gets faster, OpenCL is an advantage as it will run on multiple device platforms and for the foreseeable future.

