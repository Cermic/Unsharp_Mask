//------------------------------------------------------------------------------
//
// kernel:  blur  
//
// Purpose: Parralelise the pixel averaging method
// 
// input: out - the output pointer, in - the input pointer
// x - the x index of the current pixel, y - the y index of the current pixel
// w - the width of the image, h - the height of the image and nchannels the number of pixel channels.
//
// output: The average of the nsamples pixels within a blur radius (x,y). Pixels which
// would be outside the image, replicate the value at the image border.
// Averages the nsamples pixels within blur_radius of (x,y). Pixels which
// would be outside the image, replicate the value at the image border.

void pixel_average(
	__global unsigned char *out,
	/*__constant*/ __global const unsigned char *in,
	const int x,
	const int y,
	const int blur_radius,
	const unsigned w,
	const unsigned h,
	const unsigned nchannels)
{
	float red_total = 0, green_total = 0, blue_total = 0;

	for (int j = y - blur_radius + 1; j < y + blur_radius; ++j) {
		for (int i = x - blur_radius + 1; i < x + blur_radius; ++i) {
			const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;

			const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
			unsigned byte_offset = (r_j*w + r_i)*nchannels;
			red_total += in[byte_offset + 0];
			green_total += in[byte_offset + 1];
			blue_total += in[byte_offset + 2];
		}
	}

	// This code seems to cause a brightening effect, why is this?
	/*int i = get_global_id(0) - blur_radius + 1;
	int j = get_global_id(1) - blur_radius + 1;

	if (j < y - blur_radius + 1)
	{
		if (i < x + blur_radius)
		{
			const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;

			const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
			unsigned byte_offset = (r_j*w + r_i)*nchannels;
			red_total += in[byte_offset + 0];
			green_total += in[byte_offset + 1];
			blue_total += in[byte_offset + 2];
		}
	}*/
	const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);
	unsigned byte_offset = (y*w + x)*nchannels;
	out[byte_offset + 0] = red_total / nsamples;
	out[byte_offset + 1] = green_total / nsamples;
	out[byte_offset + 2] = blue_total / nsamples;
}

	

__kernel void blur(
	__global unsigned char* out,
	/*__constant*/ __global const unsigned char* in,
	const int blur_radius,
	const unsigned w,
	const unsigned h,
	const unsigned nchannels)
{
	//int z = get_global_size(0); // What is this value?
	int x = get_global_id(0);
	int y = get_global_id(1);
		pixel_average(out, in, x, y, blur_radius, w, h, nchannels);
}
