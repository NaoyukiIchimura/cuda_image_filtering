////
//// postprocessing_kernel.cu: the functions for postprocessing of images, which are performed on a GPU
////

///
/// The standard include files
///
#include <iostream>
#include <new>

///
/// The include files for CUDA 
///
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

///
/// The include files for thrust
///
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

///
/// The include files for postprocessing
///
#include "image_rw_cuda.h"
#include "postprocessing_kernel.h"
#include "exec_config.h"

///
/// The function for taking the absolute value of each pixel value
///
template <typename T>
__global__ void takeImageAbsoluteValueGPUKernel( T *image, const unsigned int iWidth, const unsigned int iHeight )
{

    const unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y * blockDim.y + threadIdx.y;

    if( u < iWidth && v < iHeight ) {
	unsigned int pixelPos = v * iWidth + u;
	image[pixelPos] = fabs( image[pixelPos] );
    }

}

template <typename T>
int takeImageAbsoluteValueGPU( dsaImage<T> *d_image, const unsigned int &noChannels )
{

    //
    // Set the execution configuration
    //
    const dim3 grid( iDivUp( d_image->getImageWidth(), POSTPROCESSING_BLOCK_W ), iDivUp( d_image->getImageHeight(), POSTPROCESSING_BLOCK_H ) );
    const dim3 threadBlock( POSTPROCESSING_THREAD_BLOCK_W, POSTPROCESSING_THREAD_BLOCK_H );
    
    //
    // Computing absolute values of image pixel values
    //
    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads

    if( noChannels == 3 )
	for( unsigned int i = 0; i < noChannels; i++ )
	    takeImageAbsoluteValueGPUKernel<<<grid,threadBlock>>>( d_image->getImagePtr( i ), d_image->getImageWidth(), d_image->getImageHeight() );
    else
	takeImageAbsoluteValueGPUKernel<<<grid,threadBlock>>>( d_image->getYCompPtr(), d_image->getImageWidth(), d_image->getImageHeight() );
    
    getLastCudaError( "takeImageAbsoluteValueGPUKernel() execution failed.\n" );

    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads
    
    return 0;
    
}

///
/// The function for normalizing image
///
template <typename T>
__global__ void normalizeImageGPUKernel( T *image, const unsigned int iWidth, const unsigned int iHeight, const T maxImageValue, const T minImageValue )
{

    const unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y * blockDim.y + threadIdx.y;

    if( u < iWidth && v < iHeight ) {
	const T maxPixelValue = 255.0;
	unsigned int pixelPos = v * iWidth + u;
	image[pixelPos] = maxPixelValue * ( image[pixelPos] - minImageValue ) / ( maxImageValue - minImageValue );
    }

}

template <typename T>
int normalizeImageGPU( dsaImage<T> *d_image, const unsigned int &noChannels )
{
    
    //
    // Find the maximum and minimum of the pixel value
    // The template library, thrust, is used.
    //

    //
    // Set the device pointers for the allocated image to use thrust 
    // Note that the default constructor of the thrust::device_ptr class is called by new.
    //
    thrust::device_ptr<T> *d_imageStartPtr;
    try {
	d_imageStartPtr = new thrust::device_ptr<T>[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for d_imageStartPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    
    thrust::device_ptr<T> *d_imageEndPtr;
    try {
	d_imageEndPtr = new thrust::device_ptr<T>[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for d_imageEndPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    // Since placement new is used here, the destructor of the class should be called implicitly
    // Note that the constructor with the parameter, thrust::device_ptr<T>( d_image->getImagePtr( i ) ), is called by placement new.
    if( noChannels == 3 ) { // RGB image
	for( unsigned int i = 0; i < noChannels; i++ ) {
	    new( &d_imageStartPtr[i] ) thrust::device_ptr<T>( d_image->getImagePtr( i ) );
	    d_imageEndPtr[i] = d_imageStartPtr[i] + d_image->getImageWidth() * d_image->getImageHeight();
	}
    } else if( noChannels == 1 ) { // Y Component
	new( &d_imageStartPtr[0] ) thrust::device_ptr<T>( d_image->getYCompPtr() );
	d_imageEndPtr[0] = d_imageStartPtr[0] + d_image->getImageWidth() * d_image->getImageHeight();
    }

    //
    // Find the maximum and minimum of the pixel values
    //
    thrust::device_ptr<T> *maxImageValuePtr;
    try {
	maxImageValuePtr = new thrust::device_ptr<T>[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for maxImageValuePtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    
    thrust::device_ptr<T> *minImageValuePtr;
    try {
	minImageValuePtr = new thrust::device_ptr<T>[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for minImageValuePtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    for( unsigned int i = 0; i < noChannels; i++ ) {
	maxImageValuePtr[i] = thrust::max_element( d_imageStartPtr[i], d_imageEndPtr[i] );
	minImageValuePtr[i] = thrust::min_element( d_imageStartPtr[i], d_imageEndPtr[i] );
    }

    //
    // Normalize the image
    //

    // Set the execution configuration
    const dim3 grid( iDivUp( d_image->getImageWidth(), POSTPROCESSING_BLOCK_W ), iDivUp( d_image->getImageHeight(), POSTPROCESSING_BLOCK_H ) );
    const dim3 threadBlock( POSTPROCESSING_THREAD_BLOCK_W, POSTPROCESSING_THREAD_BLOCK_H );

    // Call the kernel function
    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads

    if( noChannels == 3 )
	for( unsigned int i = 0; i < noChannels; i++ )
	    normalizeImageGPUKernel<<<grid,threadBlock>>>( d_image->getImagePtr( i ), d_image->getImageWidth(), d_image->getImageHeight(),
							   static_cast<T>(*maxImageValuePtr[i]), static_cast<T>(*minImageValuePtr[i]) );
    else
	normalizeImageGPUKernel<<<grid,threadBlock>>>( d_image->getYCompPtr(), d_image->getImageWidth(), d_image->getImageHeight(),
						       static_cast<T>(*maxImageValuePtr[0]), static_cast<T>(*minImageValuePtr[0]) );
    
    getLastCudaError( "normalizeImageGPUKernel() execution failed.\n" );

    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads

    //
    // Implicitly call the destructor for the class allocated by placement new 
    //
    if( noChannels == 3 )
	for( unsigned int i = 0; i < noChannels; i++ )
	    d_imageStartPtr[i].~device_ptr();
    else
	d_imageStartPtr[0].~device_ptr();
    
    //
    // Delete the memory spaces
    //
    delete [] minImageValuePtr;
    minImageValuePtr = 0;
    delete [] maxImageValuePtr;
    maxImageValuePtr = 0;

    delete [] d_imageEndPtr;
    d_imageEndPtr = 0;
    delete [] d_imageStartPtr;
    d_imageStartPtr = 0;

    return 0;

}

///
/// The function for level adjustment
///
template <typename T>
__global__ void adjustImageLevelGPUKernel( T *image, const unsigned int iWidth, const unsigned int iHeight, const T maxLevel )
{

    const unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y * blockDim.y + threadIdx.y;

    if( u < iWidth && v < iHeight ) {
	const T maxPixelValue = 255.0;
	unsigned int pixelPos = v * iWidth + u;
	if( image[pixelPos] > maxLevel )
	    image[pixelPos] = maxPixelValue;
	else
	    image[pixelPos] = image[pixelPos] * maxPixelValue / maxLevel;
    }

}

template <typename T>
int adjustImageLevelGPU( dsaImage<T> *d_image, const unsigned int &noChannels, const T &maxLevel )
{

    //
    // Set the execution configuration
    //
    const dim3 grid( iDivUp( d_image->getImageWidth(), POSTPROCESSING_BLOCK_W ), iDivUp( d_image->getImageHeight(), POSTPROCESSING_BLOCK_H ) );
    const dim3 threadBlock( POSTPROCESSING_THREAD_BLOCK_W, POSTPROCESSING_THREAD_BLOCK_H );

    //
    // Adjust image level
    //
    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads

    if( noChannels == 3 )
	for( unsigned int i = 0; i < noChannels; i++ )
	    adjustImageLevelGPUKernel<<<grid,threadBlock>>>( d_image->getImagePtr( i ), d_image->getImageWidth(), d_image->getImageHeight(), maxLevel );
    else
	adjustImageLevelGPUKernel<<<grid,threadBlock>>>( d_image->getYCompPtr(), d_image->getImageWidth(), d_image->getImageHeight(), maxLevel );
    
    getLastCudaError( "adjustImageLevelGPUKernel() execution failed.\n" );

    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads

    return 0;

}
    
////
//// Explicit instantication of the template functions
////
template
int takeImageAbsoluteValueGPU( dsaImage<float> *d_image, const unsigned int &noChannels );
template
int takeImageAbsoluteValueGPU( dsaImage<double> *d_image, const unsigned int &noChannels );

template
int normalizeImageGPU( dsaImage<float> *d_image, const unsigned int &noChannels );
template
int normalizeImageGPU( dsaImage<double> *d_image, const unsigned int &noChannels );

template
int adjustImageLevelGPU( dsaImage<float> *d_image, const unsigned int &noChannels, const float &maxLevel );
template
int adjustImageLevelGPU( dsaImage<double> *d_image, const unsigned int &noChannels, const double &maxLevel );
