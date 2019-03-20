//Usage: diff3dp-cuda4 natom box gpu_id k_min delta_k k_max
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <unistd.h>
#define _USE_MATH_DEFINES
#define BS 512

#define CHECK(x)\
{cudaError_t err = (x);\
if (err != cudaSuccess) {\
printf("API error failed %s:%d Returned:%d\n", __FILE__, __LINE__, err);\
exit(1); \
}\
}

void diffcore(double *coords, int natom, float box, double k_step, int ik_min, int ik_max, int iphi_max, int itheta_max);

int compare_doubles (const void *a, const void *b);

__global__ void diff_atom_loop ( double *d_coords, int natom, double k_x, double k_y, double k_z, double *d_sk_re_array, double *d_sk_im_array);

template <unsigned int blockSize> __global__ void reduce6 (double *g_idata, double *g_odata, unsigned int n);

void calculateAnisotropy(double *sk_array, int i_phi_theta, double *sk_ave, double *anisotropy);

int main (int argc, char *argv[]){ 
	FILE *infile;
	int natom, i, j, device_number;
	double *coords;

	int ik_min, ik_max, iphi_max = 36, itheta_max = 18;
	double k_resolution = 4;
	float box;
	double k_step;

	double k_min, k_max;

	natom = atoi(argv[2]);
	printf("#Natom = %i\n", natom);

	box = atof(argv[3]);
	printf("#Box = %f\n", box);

	device_number = atoi(argv[4]);
	printf("#Device ID = %i\n", device_number);
	cudaSetDevice(device_number);

	printf("#k sk_mod_avg sk_mod_max sk_mod_top1 sk_mod_top5 sk_mod_top10 sk_sq_avg sk_sq_max sk_sq_top1 sk_sq_top5 sk_sq_top10\n");
	if(argc > 5)
	{
		k_min = (double) atof(argv[5]);
		k_max = (double) atof(argv[7]);
		k_step = (double) atof(argv[6]);
		ik_min = (int) (k_min / k_step);
		ik_max = (int) (k_max / k_step);
	}
	else
	{
		ik_min = k_resolution;
		ik_max = 100;
		k_step = (double) 1. / k_resolution / box;
	}

	infile = fopen(argv[1], "rb");	

	coords = (double *)calloc(3 * natom, sizeof(double)); 	

	if( coords != NULL )
	{
		for(i=0;i<natom;i++)
		{
			for(j=0;j<3;j++)
			{
				fscanf(infile, "%lf", &coords[i*3+j]);
			}
		}
	}

	setvbuf (stdout, NULL, _IONBF, 0);
	
	diffcore( coords, natom, box, k_step, ik_min, ik_max, iphi_max, itheta_max);

	free(coords);

	return 0;
}

void diffcore(double *coords, int natom, float box, double k_step, int ik_min, int ik_max, int iphi_max, int itheta_max)
{

	int i, i1, i5, i10;
	
	int ik, iphi, itheta;
	double k, phi, theta;
	double cos_phi, sin_phi, k_cos_theta;
	double k_x, k_y, k_z;

	double *sk_re_list, *sk_im_list;
	double sk_re, sk_im, sk_sq, sk_mod, sk_sq_ave, sk_mod_ave;

	double *sk_sq_array, *sk_mod_array;
	double *sk_sq_anisotropy, *sk_mod_anisotropy;

	int i_phi_theta;

	i_phi_theta = iphi_max * itheta_max;

	double *d_coords, *d_sk_re_array, *d_sk_im_array;
	double *d_sk_re_list, *d_sk_im_list;

	CHECK(cudaMalloc(&d_coords, 3 *natom * sizeof(double)));

	CHECK(cudaMemcpy(d_coords, coords, natom * 3 * sizeof(double), cudaMemcpyHostToDevice));

	for(ik=ik_min; ik<=ik_max; ik++)
	{
		k = (double)ik * (double)k_step;

		sk_sq_array = (double *)calloc(i_phi_theta, sizeof(double));
		sk_mod_array = (double *)calloc(i_phi_theta, sizeof(double));

		for(iphi=0; iphi < iphi_max; iphi++)
		{
			phi = M_PI * 2 * (double)iphi / (double)iphi_max;
			cos_phi = (double)cos(phi);
			sin_phi = (double)sin(phi);
			for(itheta=0; itheta < itheta_max; itheta++)
			{
				theta = M_PI * ( (double)itheta / (double)itheta_max - 0.5);
				k_cos_theta = k * (double)cos(theta);
				k_x = k_cos_theta * cos_phi;
				k_y = k_cos_theta * sin_phi;
				k_z = k * (double)sin(theta);

				CHECK(cudaMalloc((void **) &d_sk_re_array, natom * sizeof(double)));
				CHECK(cudaMalloc((void **) &d_sk_im_array, natom * sizeof(double)));

				CHECK(cudaDeviceSynchronize());

				diff_atom_loop<<<natom/BS + 1, BS>>>(d_coords, natom, k_x, k_y, k_z, d_sk_re_array, d_sk_im_array);

				sk_re_list = (double *)calloc(natom/BS + 1, sizeof(double));
				sk_im_list = (double *)calloc(natom/BS + 1, sizeof(double));

				CHECK(cudaMalloc((void **) &d_sk_re_list, (natom/BS + 1) * sizeof(double)));
				CHECK(cudaMalloc((void **) &d_sk_im_list, (natom/BS + 1) * sizeof(double)));

				CHECK(cudaDeviceSynchronize());

				size_t shm_size = BS * sizeof(double);

				reduce6<(unsigned int)BS><<<natom/BS+1, BS, shm_size>>>(d_sk_re_array, d_sk_re_list, natom);
				reduce6<(unsigned int)BS><<<natom/BS+1, BS, shm_size>>>(d_sk_im_array, d_sk_im_list, natom);

				CHECK(cudaDeviceSynchronize());
	
				CHECK(cudaMemcpy(sk_re_list, d_sk_re_list, (natom/BS + 1) * sizeof(double), cudaMemcpyDeviceToHost));
				CHECK(cudaMemcpy(sk_im_list, d_sk_im_list, (natom/BS + 1) * sizeof(double), cudaMemcpyDeviceToHost));

				CHECK(cudaDeviceSynchronize());

				
				sk_re = 0;
				sk_im = 0;
				for(i=0;i<=(natom/BS);i++)
				{
					sk_re = sk_re + sk_re_list[i];
					sk_im = sk_im + sk_im_list[i];
				}

				CHECK(cudaFree(d_sk_re_array));
				CHECK(cudaFree(d_sk_im_array));

				CHECK(cudaFree(d_sk_re_list));
				CHECK(cudaFree(d_sk_im_list));

				free(sk_re_list);
				free(sk_im_list);

				sk_sq = ( sk_re * sk_re + sk_im * sk_im ) / (double)natom / (double)natom;
				sk_mod = sqrt( sk_sq );
				sk_sq_array[iphi * itheta_max + itheta] = sk_sq;
				sk_mod_array[iphi * itheta_max + itheta] = sk_mod;
			}
		}
		sk_sq_anisotropy = (double *)calloc(i_phi_theta, sizeof(double));
		sk_mod_anisotropy = (double *)calloc(i_phi_theta, sizeof(double));

		calculateAnisotropy(sk_sq_array, i_phi_theta, &sk_sq_ave, sk_sq_anisotropy);
		calculateAnisotropy(sk_mod_array, i_phi_theta, &sk_mod_ave, sk_mod_anisotropy);

//printf("#k sk_mod_avg sk_mod_max sk_mod_top1 sk_mod_top5 sk_mod_top10 sk_sq_avg sk_sq_max sk_sq_top1 sk_sq_top5 sk_sq_top10\n");
		if ((i1 = i_phi_theta / 100 - 1) < 0) i1 = 0;
		if ((i5 = i_phi_theta / 20 - 1) < 0) i5 = 0;
		if ((i10 = i_phi_theta / 10 - 1) < 0) i10 = 0;
		
		printf("%f %f %f %f %f %f %f %f %f %f %f\n", k, sk_mod_ave, sk_mod_anisotropy[0], sk_mod_anisotropy[i1], sk_mod_anisotropy[i5], sk_mod_anisotropy[i10], sk_sq_ave, sk_sq_anisotropy[0], sk_sq_anisotropy[i1], sk_sq_anisotropy[i5], sk_sq_anisotropy[i10]);

		free(sk_sq_anisotropy);
		free(sk_mod_anisotropy);

		free(sk_sq_array);
		free(sk_mod_array);
	}
	CHECK(cudaFree(d_coords));
}

int compare_doubles (const void *a, const void *b)
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}

__global__ void diff_atom_loop ( double *d_coords, int natom, double k_x, double k_y, double k_z, double *d_sk_re_array, double *d_sk_im_array)
{
	double power;
	int ib;

	double sk_re, sk_im;

	double x_a, y_a, z_a;
	double x_b, y_b, z_b;

	unsigned int ia = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int gridSize = blockDim.x * gridDim.x;

	while (ia < natom)
	{
		sk_re = 0;
		sk_im = 0;
	
		x_a = d_coords[ia * 3];
		y_a = d_coords[ia * 3 + 1];
		z_a = d_coords[ia * 3 + 2];
		for(ib = 0; ib < ia; ib++)
		{
			x_b = d_coords[ib * 3];
			y_b = d_coords[ib * 3 + 1];
			z_b = d_coords[ib * 3 + 2];
						
			power = 2 * M_PI * ( k_x * (x_a - x_b) + k_y * (y_a - y_b) + k_z * (z_a - z_b));
			sk_re += cos(power);
			sk_im += sin(power);
		}
		d_sk_re_array[ia] = sk_re;
		d_sk_im_array[ia] = sk_im;

		ia += gridSize;
	}
}

template <unsigned int blockSize> __global__ void reduce6 (double *g_idata, double *g_odata, unsigned int n)
{
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize + tid;
	unsigned int gridSize = blockSize * gridDim.x;
	sdata[tid] = 0;
	while (i < n)
	{
		sdata[tid] += g_idata[i];
		i += gridSize; 
	}
	__syncthreads();
	if (blockSize >= 512) { 
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		} __syncthreads();
	}
	if (blockSize >= 256) { 
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		} __syncthreads();
	}
	if (blockSize >= 128) { 
		if (tid <   64) {
			sdata[tid] += sdata[tid +   64];
		} __syncthreads();
	}
	if (tid < 32)
	{
		if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
		__syncthreads();
		if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
		__syncthreads();
		if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
		__syncthreads();
		if (blockSize >=    8) sdata[tid] += sdata[tid +  4];
		__syncthreads();
		if (blockSize >=    4) sdata[tid] += sdata[tid +  2];
		__syncthreads();
		if (blockSize >=    2) sdata[tid] += sdata[tid +  1];
		__syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void calculateAnisotropy(double *sk_array, int i_phi_theta, double *sk_ave, double *anisotropy)
{
		int i;
		double sk_sum = 0;

		qsort(sk_array, i_phi_theta, sizeof(double), compare_doubles);

		for(i = 1; i <= i_phi_theta; i++)
		{
			sk_sum += sk_array[i_phi_theta - i];
			anisotropy[i - 1] = sk_sum / i;
		}

		*sk_ave = sk_sum / i_phi_theta;

		for(i = 0; i < i_phi_theta; i++)
		{
			anisotropy[i] = anisotropy[i] / *sk_ave;
		}
}
