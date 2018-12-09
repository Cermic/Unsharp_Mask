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
		const char *ifilename = argc > 1 ? argv[1] :/* "../../goldhillin.ppm"; */    "../../ghost-town-8kin.ppm";    /* "../../gothicin.ppm";*//*"../../WhiteStreetIn.ppm";*/
		const char *ofilename = argc > 2 ? argv[2] :/* "../../goldhillout.ppm"; */  "../../ghost-town-8kout.ppm";    /*"../../gothicout.ppm";*/ /*"../../WhiteStreetOut.ppm";*/
		const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;

  ppm img;
  int testCaseSize = 6, testCaseIgnoreBuffer = 2;

  struct Buffers 
  {
	  std::vector<unsigned char> h_original_image, h_blurred_image, h_sharpened_image;
	  cl::Buffer d_original_image, d_sharpened_image;
	  cl::Buffer d_blurred_image1, d_blurred_image2;
  }buffers;
  
  struct ImageValues
  {
	  float alpha = 1.5f, beta = -0.5f, gamma = 0.0f;
  } imgval;

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
  // Defaults to CPU in case there are no GPUs.
  deviceSelection = CL_DEVICE_TYPE_CPU;
  for (std::vector<cl::Platform>::iterator plat = platforms.begin(); plat != platforms.end(); plat++)
  {	 
	  std::string s;
	  plat->getInfo(CL_PLATFORM_VENDOR, &s);
	  if (s == "NVIDIA Corporation")
	  {
		  deviceSelection = CL_DEVICE_TYPE_GPU;
		  std::cout << "NVIDIA GPU Selected." << std::endl;
		  break;
	  }
	  else if(s == "Advanced Micro Devices, Inc." && plat == platforms.end())
	  {
		  deviceSelection = CL_DEVICE_TYPE_GPU;
		  std::cout << "AMD GPU Selected." << std::endl;
		  break;
	  }
	
  }
  if (deviceSelection == CL_DEVICE_TYPE_CPU)
  {
	  std::cout << "CPU Selected." << std::endl;
  }
  // Create a context
  cl::Context context(deviceSelection);

  //Create cl Event
  cl::Event event;
  auto deviceList = context.getInfo<CL_CONTEXT_DEVICES>();
  //Create a program object for the context
  cl::Program program;

  std::cout << "Reading from " << ifilename << "\n" << std::endl;
  img.read(ifilename, buffers.h_original_image);

  // Allocate space for the blurred output image
  buffers.h_blurred_image.resize(img.w * img.h * img.nchannels);
  // Allocate space for the sharpened output image
  buffers.h_sharpened_image.resize(img.w * img.h * img.nchannels);

  std::cout << "Reading complete from " << ifilename << " Serial execution will now begin.\n" << std::endl;

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// Serial Execution BEGIN /////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  double serialExecutionResult = 0, serialExecutionAverage = 0;

  std::cout << "Serial process is being cycled to filter out erroneous values, please be patient... \n"<< std::endl;

  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
  {
		  auto serialExecutionPreTimer = std::chrono::steady_clock::now();

		  unsharp_mask(buffers.h_sharpened_image.data(), buffers.h_original_image.data(), blur_radius,
			  img.w, img.h, img.nchannels);

		  auto serialExecutionPostTimer = std::chrono::steady_clock::now();
		  serialExecutionResult = std::chrono::duration<double, std::ratio<1, 1000>>(serialExecutionPostTimer - serialExecutionPreTimer).count();
		  if (i >= testCaseIgnoreBuffer)
		  {
		  std::cout
			  << "Serial execution ran in "
			  << std::fixed
			  << std::setprecision(1)
			  << serialExecutionResult
			  << " milliseconds."
			  << std::endl;
		  serialExecutionAverage += serialExecutionResult;
		 }
  }
  std::cout
	  << "Serial execution average time after "
	  << testCaseSize
	  << " Iterations was "
	  << std::fixed
	  << std::setprecision(1)
	  << (serialExecutionAverage /= testCaseSize)
	  << " milliseconds.\n"
	  << std::endl;
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// Serial Execution END //////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////

   //////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// Paralllel Execution BEGIN //////////////////////////////////////
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
	  auto blur = cl::make_kernel<cl::Buffer,
								  cl::Buffer,
								  const int,
							      const unsigned,
							      const unsigned,
							      const unsigned>(program, "blur");


	  // Create Add_Weighted Kernel
	  program = cl::Program(context, util::loadProgram("../../add_weighted.cl"));
	  // Build the cl file to check for errors.
	  program.build(context.getInfo<CL_CONTEXT_DEVICES>());

	  // Create the kernel 
	  auto add_weighted = cl::make_kernel<cl::Buffer,
										  cl::Buffer,
										  const float,
										  cl::Buffer,
										  const float,
										  const float,
										  const unsigned int,
										  const unsigned int,
										  const unsigned int>(program, "add_weighted");

	  //L auto workGroupSize = add_weighted.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(cl::Device::getDefault());
	  //L auto numWorkGroups = h_original_image.size() / workGroupSize;

	  std::cout << "Parallel process is being cycled to filter out erroneous values, please be patient... \n" << std::endl;
	  //Assign buffer
	  buffers.d_original_image = cl::Buffer(context, buffers.h_original_image.begin(), buffers.h_original_image.end(), CL_MEM_READ_ONLY, true);
	  buffers.d_sharpened_image = cl::Buffer(context, buffers.h_sharpened_image.begin(), buffers.h_sharpened_image.end(), CL_MEM_READ_WRITE, true);
	  
	  for (int i = 0; i < (testCaseSize + testCaseIgnoreBuffer); i++)
	  {
			  //////////////////////////////////////////////////////////////////////////////////////////////////////
			  //////////////////////////////// Blur operation begins ///////////////////////////////////////////////
			  //////////////////////////////////////////////////////////////////////////////////////////////////////

			  parallelExecutionPreTimer = std::chrono::steady_clock::now(); // Timer before kernel execution begins

			  auto bufferAssignmentPreTimer = std::chrono::steady_clock::now();

			  //Assign buffers
			  buffers.d_blurred_image1 = cl::Buffer(context, buffers.h_blurred_image.begin(), buffers.h_blurred_image.end(), CL_MEM_READ_WRITE, true);
			  buffers.d_blurred_image2 = cl::Buffer(context, buffers.h_blurred_image.begin(), buffers.h_blurred_image.end(), CL_MEM_READ_WRITE, true);

			  auto bufferAssignmentPostTimer = std::chrono::steady_clock::now();

			  auto kernelsPreTimer = std::chrono::steady_clock::now();
		
			  // Execute Blur Kernels
				blur(
					  cl::EnqueueArgs(
					  queue,
					  cl::NDRange(img.w, img.h)),
					  buffers.d_blurred_image1,
					  buffers.d_original_image,
					  blur_radius,
				      img.w,
				      img.h,
				      img.nchannels);

				blur(
					cl::EnqueueArgs(
						queue,
						cl::NDRange(img.w, img.h)),
					buffers.d_blurred_image2,
					buffers.d_blurred_image1,
					blur_radius,
					img.w,
					img.h,
					img.nchannels);

				blur(
					cl::EnqueueArgs(
						queue,
						cl::NDRange(img.w, img.h)),
					buffers.d_blurred_image1,
					buffers.d_blurred_image2,
					blur_radius,
					img.w,
					img.h,
					img.nchannels);

			  //////////////////////////////////////////////////////////////////////////////////////////////////////
			  //////////////////////////////// Blur operation finished, now Add_Weighted ///////////////////////////
			  //////////////////////////////////////////////////////////////////////////////////////////////////////
			  // Execute Add_Weigted Kernel
			  add_weighted(
				  cl::EnqueueArgs(
					  queue,
					  cl::NDRange(img.w, img.h)),
				  buffers.d_sharpened_image,
				  buffers.d_original_image,
				  imgval.alpha,
				  buffers.d_blurred_image1,
				  imgval.beta,
				  imgval.gamma,
				  img.w,
				  img.h,
				  img.nchannels);

			  auto kernelsPostTimer = std::chrono::steady_clock::now();

			  //////////////////////////////////////////////////////////////////////////////////////////////////////
			  /////////////////// Add_Weighted finished, now copy back to host buffer for writing //////////////////
			  //////////////////////////////////////////////////////////////////////////////////////////////////////

			  // Copy the contents of d_sharpened_image to h_sharpened_image.
			  auto copyToHostPreTimer = std::chrono::steady_clock::now();
			  cl::copy(queue, buffers.d_sharpened_image, buffers.h_sharpened_image.begin(), buffers.h_sharpened_image.end());
			  auto copyToHostPostTimer = std::chrono::steady_clock::now();

			  parallelExecutionPostTimer = std::chrono::steady_clock::now(); // Timer after parallel execution is finished
			  if (i >= testCaseIgnoreBuffer)
			  {
			  parallelExecutionResult = std::chrono::duration<double, std::ratio<1, 1000>>(parallelExecutionPostTimer - parallelExecutionPreTimer).count();
			  std::cout
				  << "Total parallel execution ran in "
				  << std::fixed
				  << std::setprecision(1)
				  << parallelExecutionResult
				  << " milliseconds.\n"
				  << "Buffer assignment took "
				  << std::fixed
				  << std::setprecision(1)
				  << std::chrono::duration<double,std::ratio<1,1000>>(bufferAssignmentPostTimer - bufferAssignmentPreTimer).count()
				  << " milliseconds.\n"
				  << "Kernels took "
				  << std::fixed
				  << std::setprecision(1)
				  << std::chrono::duration<double, std::ratio<1, 1000>>(kernelsPostTimer - kernelsPreTimer).count()
				  << " milliseconds.\n"
				  << "Copying back to host took "
				  << std::chrono::duration<double, std::ratio<1, 1000>>(copyToHostPostTimer - copyToHostPreTimer).count()
				  << " milliseconds.\n"
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
		  << std::fixed
		  << std::setprecision(1)
		  << (parallelExecutionAverage /= testCaseSize)
		  << " milliseconds.\n"
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
  std::cout
	  << "Parallel execution was "
	  << std::fixed
	  << std::setprecision(1)
	  << speedFactorDifference 
	  << " Times faster than Serial execution \n" << std::endl;

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Paralllel Execution END ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

  // Write the sharpened image - to become the new picture.
  std::cout << "Writing final image to " << ofilename << "\n" << std::endl;

  img.write(ofilename, buffers.h_sharpened_image);

  std::cout << "Writing complete to " << ofilename << ".\n" << std::endl;

  system("pause");
  return 0;
}

