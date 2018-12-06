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
	const unsigned nchannels
	/*__local unsigned char *localData1*/
	/*__local unsigned char *localData2*/)
{
		int x = get_global_id(0);
		int y = get_global_id(1);

		//barrier(CLK_GLOBAL_MEM_FENCE);

		//int l = get_local_size(0);
		//int lx1 = get_local_id(0);
		//int ly1 = get_local_id(1);

		//int lx2 = get_local_id(0);
		//int ly2 = get_local_id(1);

		//localData1[lx1] = in1[x];
		//localData1[ly1] = in1[y];

		//localData2[lx2] = in2[x];
		//localData2[ly2] = in2[y];
		//barrier(CLK_LOCAL_MEM_FENCE);

		// Do I need 2 local data buffers for this since there are 2 in buffers?
			unsigned byte_offset = (y*w + x)*nchannels;

			float tmp = in1[byte_offset + 0] * alpha + in2[byte_offset + 0] * beta + gamma;
			out[byte_offset + 0] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

			tmp = in1[byte_offset + 1] * alpha + in2[byte_offset + 1] * beta + gamma;
			out[byte_offset + 1] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

			tmp = in1[byte_offset + 2] * alpha + in2[byte_offset + 2] * beta + gamma;
			out[byte_offset + 2] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

	/*		if (lx1 == 0)
				out[get_group_id(0) = localData1[0];
			if (ly1 == 0)
				out[get_group_id(1) = localData1[1];

			if (lx2 == 0)
				out[get_group_id(0) = localData2[0];
			if (ly2 == 0)
				out[get_group_id(1) = localData2[1];*/
		

}
