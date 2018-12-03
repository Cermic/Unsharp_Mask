#define __CL_ENABLE_EXCEPTIONS

#include <chrono>
#include "unsharp_mask.hpp"
#include "CL/cl.hpp"
#include "CL/err_code.h"
#include "CL/util.hpp" // utility library
#include <iomanip>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.

int main(int argc, char *argv[])
{
  const char *ifilename = argc > 1 ? argv[1]			: /*"../../goldhillin.ppm";*/ "../../ghost-town-8kin.ppm"; 
  const char *ofilename = argc > 2 ?           argv[2]  : /*"../../goldhillout.ppm";*/ "../../ghost-town-8kout.ppm";
  const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;

  ppm img;
  std::vector<unsigned char> h_original_image, h_blurred_image, h_sharpened_image;

  cl::Buffer d_original_image, d_blurred_image, d_sharpened_image; // What do I allocate for sharpened image?
  // See if the logic for the kernels is right.
  std::cout << "Reading from " << ifilename << "\n" << std::endl;
  img.read(ifilename, h_original_image);

  // Does any of this need timed?


  // Allocate space for the blurred output image
  h_blurred_image.resize(img.w * img.h * img.nchannels);
  // Allocate space for the sharpened output image
  h_sharpened_image.resize(img.w * img.h * img.nchannels);

  // Time it?

  // Create a context
  cl::Context context(DEVICE);

  //Create cl Event
  cl::Event event;
  auto deviceList = context.getInfo<CL_CONTEXT_DEVICES>();
  //Create a program object for the context
  cl::Program program;

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// Serial Execution BEGIN /////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  auto serialExecutionPreTimer = std::chrono::steady_clock::now();

  unsharp_mask(h_sharpened_image.data(), h_original_image.data(), blur_radius,
	  img.w, img.h, img.nchannels);

  auto serialExecutionPostTimer = std::chrono::steady_clock::now();
  std::cout 
	  << "Serial execution ran in "
	  << std::chrono::duration<double>(serialExecutionPostTimer - serialExecutionPreTimer).count() 
	  << " seconds.\n"
	  << std::endl;

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// Serial Execution END //////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////

  // Placeholders for the Parallel Timers.
  std::chrono::time_point<std::chrono::steady_clock> parallelExecutionPreTimer, parallelExecutionPostTimer;
  try
  {
	  //////////////////////////////////////////////////////////////////////////////////////////////////////
	  //////////////////////////////// Blur operation begins ///////////////////////////////////////////////
	  //////////////////////////////////////////////////////////////////////////////////////////////////////

	  parallelExecutionPreTimer = std::chrono::steady_clock::now(); // Timer before kernel execution begins

	  // Load Kernel Source
	  program = cl::Program(context, util::loadProgram("../../blur.cl"));
	  // Build the cl file to check for errors.
	  program.build(context.getInfo<CL_CONTEXT_DEVICES>());
	  // Get the command queue
	  cl::CommandQueue queue(context);

	  // Create the kernel
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

	  auto blurPreTimer = std::chrono::steady_clock::now(); // Timer before blur launch

	  queue.enqueueNDRangeKernel(
		  blur, 
		  cl::NullRange, 
		  cl::NDRange(img.w, img.h), 
		  cl::NullRange, 
		  NULL, 
		  &event);
	  auto blurPostTimer = std::chrono::steady_clock::now();  // Timer after blur finish

	  std::cout 
		  << "Blur kernel ran in "
		  << std::fixed
		  << std::setprecision(7)
		  << std::chrono::duration<double>(blurPostTimer - blurPreTimer).count() 
		  << " seconds.\n"
		  << std::endl;

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
	  // Since this is empty space it could just be allocated as such instead of being copied here?

	  add_weighted.setArg(0, d_sharpened_image);
	  add_weighted.setArg(1, d_original_image);
	  add_weighted.setArg(2, alpha);
	  add_weighted.setArg(3, d_blurred_image);
	  add_weighted.setArg(4, beta);
	  add_weighted.setArg(5, gamma);
	  add_weighted.setArg(6, img.w);
	  add_weighted.setArg(7, img.h);
	  add_weighted.setArg(8, img.nchannels);

	  auto add_weightedPreTimer = std::chrono::steady_clock::now(); // Timer before add_weighted launch

	  queue.enqueueNDRangeKernel(
		  add_weighted,
		  cl::NullRange,
		  cl::NDRange(img.w, img.h),
		  cl::NullRange,
		  NULL,
		  &event);

	  auto add_weightedPostTimer = std::chrono::steady_clock::now(); // Timer after add_weighted finish

	  std::cout
		  << "Add_Weighted kernel ran in "
		  << std::fixed
		  << std::setprecision(7)
		  << std::chrono::duration<double>(add_weightedPostTimer - add_weightedPreTimer).count()
		  << " seconds.\n"
		  << std::endl;

	 queue.finish();
	  //////////////////////////////////////////////////////////////////////////////////////////////////////
	  /////////////////// Add_Weighted finished, now copy back to host buffer for writing //////////////////
	  //////////////////////////////////////////////////////////////////////////////////////////////////////

	  // Copy the contents of d_sharpened_image to h_sharpened_image.
	  cl::copy(queue, d_sharpened_image, h_sharpened_image.begin(), h_sharpened_image.end());

	  parallelExecutionPostTimer = std::chrono::steady_clock::now(); // Timer after kernel execution is finished
	  std::cout
		  << "Parallel execution ran in "
		  << std::chrono::duration<double>(parallelExecutionPostTimer - parallelExecutionPreTimer).count()
		  << " seconds.\n"
		  << std::endl;
	  // Does this need to be timed also or just the kernel execution?
	  
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

  // Write the sharpened image - to become the new picture.
  std::cout << "Writing final image to " << ofilename << "\n" << std::endl;

  img.write(ofilename, h_sharpened_image);

  // Factor by which Parallel execution was faster than Serial execution.
  //double speedFactorDifference = std::chrono::duration<double>(serialExecutionPostTimer / parallelExecutionPostTimer).count();
  //// How to do this without running into this problem? How do I grab the count values out of post of these to divide them?
  //std::cout << "Parallel execution was " << speedFactorDifference << " Times faster than Serial execution" << std::endl;

  system("pause");
  return 0;
}

