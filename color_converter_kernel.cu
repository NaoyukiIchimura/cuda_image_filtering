////
//// color_converter_kernel.cu: the member and kernel functions for the colorConverter class
////

///
/// The standard include file
///
#include <iostream>

///
/// The include files for CUDA
///
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

///
/// The include files for the color converter
///
#include "image_rw_cuda.h"
#include "color_converter.h"
#include "color_converter_kernel.h"
#include "exec_config.h"

///
/// The kernel function for calculating Y component
/// Note that a __global__ routine cannot have reference arguments.
///
template <typename T>
__global__ void calYComponentGPUKernel( const T *rImage, const T *gImage, const T *bImage,
					const int iWidth, const int iHeight,
					T *YImage )
{

    const unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y * blockDim.y + threadIdx.y;

    if( u < iWidth && v < iHeight ) {
	unsigned int pixelPos = v * iWidth + u;
	YImage[pixelPos] = 0.2126 * rImage[pixelPos] + 0.7152 * gImage[pixelPos] + 0.0722 * bImage[pixelPos];
    }

}

///
/// The member function for launching the kernel function for calculating Y component
///
template <typename T>
int colorConverter<T>::calYComponentGPU( dsaImage<T> *d_image )
{
    

    //
    // Set the execution configuration
    //
    const dim3 grid( iDivUp( d_image->getImageWidth(), Y_COMP_BLOCK_W ), iDivUp( d_image->getImageHeight(), Y_COMP_BLOCK_H ) ); 
    const dim3 threadBlock( Y_COMP_THREAD_BLOCK_W, Y_COMP_THREAD_BLOCK_H );

    //
    // Calculating Y component by a GPU
    //
    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads
    
    calYComponentGPUKernel<<<grid,threadBlock>>>( d_image->getImagePtr( 0 ), d_image->getImagePtr( 1 ), d_image->getImagePtr( 2 ),
						  d_image->getImageWidth(), d_image->getImageHeight(),
						  d_image->getYCompPtr() );
    
    getLastCudaError( "CalYComponentGPUKernel() execution failed.\n" );

    checkCudaErrors( cudaDeviceSynchronize() );	// wait for threads

    return 0;

}

////
//// Explicit instantiation of the template class
////
template class colorConverter<float>;
template class colorConverter<double>;
