#define __CL_ENABLE_EXCEPTIONS

#include <chrono>
#include "unsharp_mask.hpp"
#include "CL/cl.hpp"
#include "CL/err_code.h"
#include "CL/util.hpp" // utility library
#include <iomanip>

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.

int main(int argc, char *argv[])
{
  const char *ifilename = argc > 1 ?		   argv[1]  : "../../goldhillin.ppm"; /*"../../ghost-town-8kin.ppm"; */
  const char *ofilename = argc > 2 ?           argv[2]  : "../../goldhillout.ppm"; /*"../../ghost-town-8kout.ppm";*/
  const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;

  ppm img;
  int testCaseSize = 6, testCaseIgnoreBuffer = 1;

  std::vector<unsigned char> h_original_image, h_blurred_image, h_sharpened_image;
  cl::Buffer d_original_image, d_blurred_image, d_sharpened_image; 

  // Discover number of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::cout << "\nNumber of OpenCL plaforms: " << platforms.size() << std::endl;

  // Investigate each platform
  std::cout << "\n-------------------------" << std::endl;
  for (std::vector<cl::Platform>::iterator plat = platforms.begin(); plat != platforms.end(); plat++)
  {
	  std::string s;
	  plat->getInfo(CL_PLATFORM_NAME, &s);
	  std::cout << "Platform: " << s << std::endl;

	  plat->getInfo(CL_PLATFORM_VENDOR, &s);
	  std::cout << "\tVendor:  " << s << std::endl;

	  plat->getInfo(CL_PLATFORM_VERSION, &s);
	  std::cout << "\tVersion: " << s << std::endl;

	  // Discover number of devices
	  std::vector<cl::Device> devices;
	  plat->getDevices(CL_DEVICE_TYPE_ALL, &devices);
	  std::cout << "\n\tNumber of devices: " << devices.size() << std::endl;

	  // Investigate each device
	  for (std::vector<cl::Device>::iterator dev = devices.begin(); dev != devices.end(); dev++)
	  {
		  std::cout << "\t-------------------------" << std::endl;

		  dev->getInfo(CL_DEVICE_NAME, &s);
		  std::cout << "\t\tName: " << s << std::endl;

		  dev->getInfo(CL_DEVICE_OPENCL_C_VERSION, &s);
		  std::cout << "\t\tVersion: " << s << std::endl;

		  int i;
		  dev->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &i);
		  std::cout << "\t\tMax. Compute Units: " << i << std::endl;

		  size_t size;

		  dev->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
		  std::cout << "\t\tMax Work-group Total Size: " << size << std::endl;

		  std::vector<size_t> d;
		  dev->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &d);
		  std::cout << "\t\tMax Work-group Dims: (";
		  for (std::vector<size_t>::iterator st = d.begin(); st != d.end(); st++)
			  std::cout << *st << " ";
		  std::cout << "\x08)" << std::endl;

		  std::cout << "\t-------------------------" << std::endl;
	  }

	  std::cout << "\n-------------------------\n";
 }

  // Device Selection - Prefers GPUs
  int deviceSelection;
  // Defaults to CPU incase there are no GPUs.
  deviceSelection = CL_DEVICE_TYPE_CPU;
  for (std::vector<cl::Platform>::iterator plat = platforms.begin(); plat != platforms.end(); plat++)
  {
	  std::string s;
	  plat->getInfo(CL_PLATFORM_VENDOR, &s);
	  if (s == "NVIDIA Corporation")
	  {
		  deviceSelection = CL_DEVICE_TYPE_GPU;
		  break;
	  }
	  else if(s == "Advanced Micro Devices, Inc.")
	  {
		  deviceSelection = CL_DEVICE_TYPE_GPU;
	  }
  }

  // Create a context
  cl::Context context(deviceSelection);

  //Create cl Event
  cl::Event event;
  auto deviceList = context.getInfo<CL_CONTEXT_DEVICES>();
  //Create a program object for the context
  cl::Program program;

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// Serial Execution BEGIN /////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  double serialExecutionResult = 0, serialExecutionAverage = 0;

  std::cout << "Reading from " << ifilename << "\n" << std::endl;
  img.read(ifilename, h_original_image);
  std::cout << "Reading complete from " << ifilename << ".\n" << std::endl;

  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
  {
		 // auto serialExecutionPreTimer = std::chrono::steady_clock::now();

		  // Allocate space for the blurred output image
		  h_blurred_image.resize(img.w * img.h * img.nchannels);
		  // Allocate space for the sharpened output image
		  h_sharpened_image.resize(img.w * img.h * img.nchannels);

		  auto serialExecutionPreTimer = std::chrono::steady_clock::now();

		  unsharp_mask(h_sharpened_image.data(), h_original_image.data(), blur_radius,
			  img.w, img.h, img.nchannels);

		  auto serialExecutionPostTimer = std::chrono::steady_clock::now();
		  serialExecutionResult = std::chrono::duration<double>(serialExecutionPostTimer - serialExecutionPreTimer).count();
		  if (i >= testCaseIgnoreBuffer)
		  {
		  std::cout
			  << "Serial execution ran in "
			  << serialExecutionResult
			  << " seconds."
			  << std::endl;
		  serialExecutionAverage += serialExecutionResult;
		 }
  }
  std::cout
	  << "Serial execution average time after "
	  << testCaseSize
	  << " Iterations was "
	  << (serialExecutionAverage / testCaseSize)
	  << " seconds.\n"
	  << std::endl;
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// Serial Execution END //////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////

  // Placeholders for the Parallel Timers.
  std::chrono::time_point<std::chrono::steady_clock> parallelExecutionPreTimer, parallelExecutionPostTimer;
  double parallelExecutionResult = 0, parallelExecutionAverage =0;
  try
  {
	  // Get the command queue
	  cl::CommandQueue queue(context);

	  // Load Kernel Source
	  program = cl::Program(context, util::loadProgram("../../blur.cl"));
	  // Build the cl file to check for errors.
	  program.build(context.getInfo<CL_CONTEXT_DEVICES>());
	  // Create the kernel
	  cl::Kernel blur = cl::Kernel(program, "blur");

	  // Create Add_Weighted Kernel
	  program = cl::Program(context, util::loadProgram("../../add_weighted.cl"));
	  // Build the cl file to check for errors.
	  program.build(context.getInfo<CL_CONTEXT_DEVICES>());
	  float alpha = 1.5f; float beta = -0.5f; float gamma = 0.0f;

	  // Create the kernel 
	  cl::Kernel add_weighted = cl::Kernel(program, "add_weighted");

	 /* auto workGroupSize = add_weighted.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>();
	  auto numWorkGroups = h_original_image.size() / workGroupSize;*/

	  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
	  {
			  //////////////////////////////////////////////////////////////////////////////////////////////////////
			  //////////////////////////////// Blur operation begins ///////////////////////////////////////////////
			  //////////////////////////////////////////////////////////////////////////////////////////////////////

			  parallelExecutionPreTimer = std::chrono::steady_clock::now(); // Timer before kernel execution begins

			  //Assign buffers
			  d_original_image = cl::Buffer(context, h_original_image.begin(), h_original_image.end(), CL_MEM_READ_WRITE, true);
			  d_blurred_image = cl::Buffer(context, h_blurred_image.begin(), h_blurred_image.end(), CL_MEM_READ_WRITE, true);

			  // Set Kernel Arguments
			  blur.setArg(0, d_blurred_image);
			  blur.setArg(1, d_original_image);
			  blur.setArg(2, blur_radius);
			  blur.setArg(3, img.w);
			  blur.setArg(4, img.h);
			  blur.setArg(5, img.nchannels);

			  queue.enqueueNDRangeKernel(
				  blur,
				  cl::NullRange,
				  cl::NDRange(img.w, img.h),
				  cl::NullRange,
				  NULL,
				  &event);

			  //////////////////////////////////////////////////////////////////////////////////////////////////////
			  //////////////////////////////// Blur operation finished, now Add_Weighted ///////////////////////////
			  //////////////////////////////////////////////////////////////////////////////////////////////////////

			  //Assign buffer
			  d_sharpened_image = cl::Buffer(context, h_sharpened_image.begin(), h_sharpened_image.end(), CL_MEM_READ_WRITE, true);

			 // d_sharpened_image = cl::Buffer(context, sizeof(insigned char) * numWorkGroups, CL_MEM_READ_WRITE, true);
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
			  //add_weighted.setArg(9, sizeof(unsigned char) * workGroupSize, nullptr);
			  //add_weighted.setArg(10, sizeof(unsigned char) * workGroupSize, nullptr);

			  queue.enqueueNDRangeKernel(
				  add_weighted,
				  cl::NullRange,
				  cl::NDRange(img.w, img.h),
				  cl::NullRange /* cl::NDRange(workGroupSize)*/,
				  NULL,
				  &event);
			   // Work group size must be a multiple of the global NDRange
			  queue.finish();
			  //////////////////////////////////////////////////////////////////////////////////////////////////////
			  /////////////////// Add_Weighted finished, now copy back to host buffer for writing //////////////////
			  //////////////////////////////////////////////////////////////////////////////////////////////////////

			  // Copy the contents of d_sharpened_image to h_sharpened_image.
			  cl::copy(queue, d_sharpened_image, h_sharpened_image.begin(), h_sharpened_image.end());

			  parallelExecutionPostTimer = std::chrono::steady_clock::now(); // Timer after parallel execution is finished
			  if (i >= testCaseIgnoreBuffer)
			  {
			  parallelExecutionResult = std::chrono::duration<double>(parallelExecutionPostTimer - parallelExecutionPreTimer).count();
			  std::cout
				  << "Parallel execution ran in "
				  << parallelExecutionResult
				  << " seconds."
				  << std::endl;
			  parallelExecutionAverage += parallelExecutionResult;

			  // Test the results
			  int correct = 0;
			}
	  }
	  std::cout
		  << "Parallel execution average time after "
		  << testCaseSize
		  << " Iterations was "
		  << (parallelExecutionAverage / testCaseSize)
		  << " seconds.\n"
		  << std::endl;
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

  // Factor by which Parallel execution was faster than Serial execution.
  double speedFactorDifference = (serialExecutionAverage /= parallelExecutionAverage);
  std::cout << "Parallel execution was " << speedFactorDifference << " Times faster than Serial execution \n" << std::endl;

  // Write the sharpened image - to become the new picture.
  std::cout << "Writing final image to " << ofilename << "\n" << std::endl;

  img.write(ofilename, h_sharpened_image);

  std::cout << "Writing complete." << ofilename << "\n" << std::endl;

  system("pause");
  return 0;
}

