#define __CL_ENABLE_EXCEPTIONS

#include <chrono>
#include "unsharp_mask.hpp"
#include "CL/cl.hpp"
#include "CL/err_code.h"
#include "CL/util.hpp" // utility library

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.

int main(int argc, char *argv[])
{
  const char *ifilename = argc > 1 ?           argv[1] : "../../goldhillin.ppm"; /*../../ghost-town-8k.ppm";*/
  const char *ofilename = argc > 2 ?           argv[2] : "../../goldhillout.ppm";
  const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;

  ppm img;
  std::vector<unsigned char> h_original_image, h_blurred_image, h_sharpened_image;

  cl::Buffer d_original_image, d_blurred_image, d_sharpened_image; // What do I allocate for sharpened image?
  // See if the logic for the kernels is right.
  std::cout << "Reading from " << ifilename << std::endl;
  img.read(ifilename, h_original_image);
  // Allocate space for the blurred output image
  h_blurred_image.resize(img.w * img.h * img.nchannels);
  // Allocate space for the sharpened output image
  h_sharpened_image.resize(img.w * img.h * img.nchannels);

  auto t1 = std::chrono::steady_clock::now();

  // Create a context
  cl::Context context(DEVICE);

  //Create cl Event
  cl::Event event;
  auto deviceList = context.getInfo<CL_CONTEXT_DEVICES>();
  //Create a program object for the context
  cl::Program program;
  try
  {
	  //////////////////////////////////////////////////////////////////////////////////////////////////////
	  //////////////////////////////// Blur operation begins ///////////////////////////////////////////////
	  //////////////////////////////////////////////////////////////////////////////////////////////////////

	  // Load Kernel Source
	  program = cl::Program(context, util::loadProgram("../../blur.cl"));
	  // Build the cl file to check for errors.
	  program.build(context.getInfo<CL_CONTEXT_DEVICES>());
	  // Get the command queue
	  cl::CommandQueue queue(context);

	  // Create the kernel
	  //cl::make_kernel<cl::Buffer, cl::Buffer, int, int, int, int> blur(program, "blur");
	  cl::Kernel blur = cl::Kernel(program, "blur");
	  
	  //Assign buffers
	  d_original_image = cl::Buffer(context, h_original_image.begin(), h_original_image.end(), CL_MEM_READ_WRITE, true);
	  d_blurred_image = cl::Buffer(context, h_blurred_image.begin(), h_blurred_image.end(), CL_MEM_READ_WRITE, true);
	  
	  // Last parameter being false makes no difference here at the moment.
	  //d_original_image = cl::Buffer(context, CL_MEM_READ_WRITE, h_original_image.begin(), h_original_image.end(), true);
	 //d_blurred_image = cl::Buffer(context, CL_MEM_READ_WRITE, h_blurred_image.begin(), h_blurred_image.end(), true);

	  // Set Kernel Arguments
	  blur.setArg(0, d_blurred_image);
	  blur.setArg(1, d_original_image);
	  blur.setArg(2, blur_radius);
	  blur.setArg(3, img.w);
	  blur.setArg(4, img.h);
	  blur.setArg(5, img.nchannels);

	  // Should I be using h_data_sharp in this kernel portion or h_data_out?

	  util::Timer timer1;

	  queue.enqueueNDRangeKernel(
		  blur, 
		  cl::NullRange, 
		  cl::NDRange(img.w, img.h), 
		  cl::NullRange, 
		  NULL, 
		  &event);

	  double rtime1 = static_cast<double>(timer1.getTimeMilliseconds()) / 1000.0;
	  printf("\nThe kernels ran in %lf seconds\n", rtime1);

	  //////////////////////////////////////////////////////////////////////////////////////////////////////
	  //////////////////////////////// Blur operation finished, now Add_Weighted ///////////////////////////
	  //////////////////////////////////////////////////////////////////////////////////////////////////////

	  program = cl::Program(context, util::loadProgram("../../add_weighted.cl"));
	  // Build the cl file to check for errors.
	  program.build(context.getInfo<CL_CONTEXT_DEVICES>());
	  float alpha = 1.5f; float beta = -0.5f; float gamma = 0.0f;

	  // Create the kernel 
	  cl::Kernel add_weighted = cl::Kernel(program, "add_weighted");

	  //Assign buffer
	  d_sharpened_image = cl::Buffer(context, h_sharpened_image.begin(), h_sharpened_image.end(), CL_MEM_READ_WRITE, true);

	  add_weighted.setArg(0, d_sharpened_image);
	  add_weighted.setArg(1, d_original_image);
	  add_weighted.setArg(2, alpha);
	  add_weighted.setArg(3, d_blurred_image);
	  add_weighted.setArg(4, beta);
	  add_weighted.setArg(5, gamma);
	  add_weighted.setArg(6, img.w);
	  add_weighted.setArg(7, img.h);
	  add_weighted.setArg(8, img.nchannels);

	  queue.enqueueNDRangeKernel(
		  add_weighted,
		  cl::NullRange,
		  cl::NDRange(img.w, img.h),
		  cl::NullRange,
		  NULL,
		  &event);

	  util::Timer timer2;

	 queue.finish();

	  double rtime2 = static_cast<double>(timer2.getTimeMilliseconds()) / 1000.0;
	  printf("\nThe kernels ran in %lf seconds\n", rtime2);

	  //////////////////////////////////////////////////////////////////////////////////////////////////////
	  /////////////////// Add_Weighted finished, now copy back to host buffer for writing //////////////////
	  //////////////////////////////////////////////////////////////////////////////////////////////////////

	  // Copy the contents of d_sharpened_image to h_sharpened_image.
	  cl::copy(queue, d_sharpened_image, h_sharpened_image.begin(), h_sharpened_image.end());
	  
	  // Test the results
	  int correct = 0;
	}
  catch (cl::Error err ) {
	  if (err.err() == CL_BUILD_PROGRAM_FAILURE)
	  {
		  for (cl::Device dev : deviceList)
		  {
			  // Check the build status
			  cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
			  if (status != CL_BUILD_ERROR)
				  continue;

			  // Get the build log
			  std::string name = dev.getInfo<CL_DEVICE_NAME>();
			  std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			  std::cerr << "Build log for " << name << ":" << std::endl
				  << buildlog << std::endl;
		  }
	  }
	  else
	  {
		  std::cout << "Exception\n";
		  std::cerr
			  << "ERROR: "
			  << err.what()
			  << "("
			  << err_code(err.err())
			  << ")"
			  << std::endl;
	  }
  }

  auto t2 = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration<double>(t2-t1).count() << " seconds.\n";

  // Write the sharpened image - to become the new picture.
  std::cout << "Writing final image to " << ofilename << std::endl;

  img.write(ofilename, h_sharpened_image);
  
  return 0;
}

