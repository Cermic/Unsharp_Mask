//------------------------------------------------------------------------------
//
// kernel:  add_weighted  
//
// Purpose: Adds the blurred image data into the original, producing the sharpened image.
// 
// input: out - the sharpened image, in1 - the original image, in2 - the blurred image.
// alpha, beta & gamma - weighting values for the unsharpening calculation. 
// w - the width of the image, h - the height of the image and nchannels the number of pixel channels.
//
// output: Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma) 
// and returns the sharpened image.

	__kernel void add_weighted(
	__global unsigned char *out,
	__global const unsigned char *in1, 
	const float alpha,
	__global const unsigned char *in2,
	const float  beta, 
	const float gamma,
	const unsigned w, 
	const unsigned h, 
	const unsigned nchannels)
{
		int x = get_global_id(0);
		int y = get_global_id(1);

			unsigned byte_offset = (y*w + x)*nchannels;

			float tmp = in1[byte_offset + 0] * alpha + in2[byte_offset + 0] * beta + gamma;
			out[byte_offset + 0] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

			tmp = in1[byte_offset + 1] * alpha + in2[byte_offset + 1] * beta + gamma;
			out[byte_offset + 1] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

			tmp = in1[byte_offset + 2] * alpha + in2[byte_offset + 2] * beta + gamma;
			out[byte_offset + 2] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;
}
