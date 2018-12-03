// Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma)

// Might need to define UCHAR_MAX here - not sure if headers can be included .
//template <typename T>
	__kernel void add_weighted(
	__global unsigned char *out,
	/*__constant*/ __global const unsigned char *in1, 
	const float alpha,
	/*__constant*/ __global const unsigned char *in2,
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
