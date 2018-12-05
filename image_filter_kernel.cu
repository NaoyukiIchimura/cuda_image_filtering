////
//// The kernel functions for an image filter
////

///
/// The standard include files
///
#include <iostream>

#include <cstdlib>
#include <cmath>

///
/// The include files for CUDA
///
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

///
/// The include files for an image filter
///
#include "image_rw_cuda.h"
#include "exec_config.h"
#include "image_filter.h"
#include "image_filter_kernel.h"
#include "constmem_type.h"

////
//// The member function of the imageFilter class and the kernel function associated with it
////

///
/// A filter kernel in constant memory
/// A constant variable must be in the global scope.
/// The type of constant memory is defined in constmem_type.h, because I couldn't find out
/// the way to set the type of the constant memory using template.
/// Note that the size of the array is set in image_filter_kernel.h.
///
__device__ __constant__ CONSTMEM_TYPE d_cKernel[ MAX_KERNEL_SIZE * MAX_KERNEL_SIZE ];

///
/// The kernel for an image filter
/// The references cannot be used for passing arguments.
///
template <typename T>
__global__ void applyFilterGPUKernel( const T *d_iImage, const unsigned int iWidth, const unsigned int iHeight,
				      const unsigned int blockWidth, const unsigned int blockHeight, const int hKernelSize,
				      T *d_oImage, const unsigned int oWidth, const unsigned int oHeight )
{

    //
    // Note that blockDim.(x,y) cannot be used instead of blockWidth and blockHeight,
    // because the size of the thread block is not equal to the size of the block
    // due to the apron and the use of sub-blocks.
    //
    
    //
    // Set the size of a tile
    //
    const unsigned int tileWidth = blockWidth + 2 * hKernelSize;
    const unsigned int tileHeight = blockHeight + 2 * hKernelSize;

    // 
    // Set the number of sub-blocks in a block
    //
    const unsigned int noSubBlocks = static_cast<unsigned int>(ceil( static_cast<double>(tileHeight)/static_cast<double>(blockDim.y) ));

    //
    // Set the start position of the block, which is determined by blockIdx. 
    // Note that since padding is applied to the input image, the origin of the block is ( hKernelSize, hKernelSize )
    //
    const unsigned int blockStartCol = blockIdx.x * blockWidth + hKernelSize;
    const unsigned int blockEndCol = blockStartCol + blockWidth;

    const unsigned int blockStartRow = blockIdx.y * blockHeight + hKernelSize;
    const unsigned int blockEndRow = blockStartRow + blockHeight;

    //
    // Set the position of the tile which includes the block and its apron
    //
    const unsigned int tileStartCol = blockStartCol - hKernelSize;
    const unsigned int tileEndCol = blockEndCol + hKernelSize;
    const unsigned int tileEndClampedCol = min( tileEndCol, iWidth );

    const unsigned int tileStartRow = blockStartRow - hKernelSize;
    const unsigned int tileEndRow = blockEndRow + hKernelSize;
    const unsigned int tileEndClampedRow = min( tileEndRow, iHeight );

    //
    // Set the size of the filter kernel
    //
    const unsigned int kernelSize = 2 * hKernelSize + 1;

    //
    // Shared memory for the tile
    //
    __shared__ T sData[IMAGE_FILTER_MAX_SHARED_MEMORY_SIZE];

    //
    // Copy the tile into shared memory
    //
    unsigned int tilePixelPosCol = threadIdx.x;
    unsigned int iPixelPosCol = tileStartCol + tilePixelPosCol;
    for( unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++ ) {

	unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
	unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;

	if( iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow ) { // Check if the pixel in the image
	    unsigned int iPixelPos = iPixelPosRow * iWidth + iPixelPosCol;
	    unsigned int tilePixelPos = tilePixelPosRow * tileWidth + tilePixelPosCol;
	    sData[tilePixelPos] = d_iImage[iPixelPos];
	}
	
    }

    //
    // Wait for all the threads for data loading
    //
    __syncthreads();

    //
    // Perform convolution
    //
    tilePixelPosCol = threadIdx.x;
    iPixelPosCol = tileStartCol + tilePixelPosCol;
    for( unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++ ) {

	unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
	unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;

	// Check if the pixel in the tile and image.
	// Note that the apron of the tile is excluded.
	if( iPixelPosCol >= tileStartCol + hKernelSize && iPixelPosCol < tileEndClampedCol - hKernelSize &&
	    iPixelPosRow >= tileStartRow + hKernelSize && iPixelPosRow < tileEndClampedRow - hKernelSize ) {

	    // Compute the pixel position for the output image
	    unsigned int oPixelPosCol = iPixelPosCol - hKernelSize; // removing the origin
	    unsigned int oPixelPosRow = iPixelPosRow - hKernelSize;
	    unsigned int oPixelPos = oPixelPosRow * oWidth + oPixelPosCol;

	    unsigned int tilePixelPos = tilePixelPosRow * tileWidth + tilePixelPosCol;

	    d_oImage[oPixelPos] = 0.0;
	    for( int i = -hKernelSize; i <= hKernelSize; i++ ) {
		for( int j = -hKernelSize; j <= hKernelSize; j++ ) {
		    int tilePixelPosOffset = i * tileWidth + j;
		    int coefPos = ( i + hKernelSize ) * kernelSize + ( j + hKernelSize );
		    d_oImage[oPixelPos] += sData[ tilePixelPos + tilePixelPosOffset ] * d_cKernel[coefPos];
		}
	    }

	}
	
    }

}

///
/// The member function launching the kernel function
///
template <typename T>
int imageFilter<T>::applyFilterGPU( dsaImage<T> &d_iImage, const unsigned int &noChannels, dsaImage<T> *d_oImage) const
{

    //
    // Copy the filter kernel to constant memory of a device
    //

    // Check the size of the constant memory to be used
    // The size of the constant memory of a GPU defined in exec_config.h
    if( MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * sizeof(T) > CONSTANT_MEMORY_SIZE ) {
	std::cerr << "The maximum size of a image filter, "
		  << MAX_KERNEL_SIZE
		  << ", is too large for the GPU: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }


    // Check the size of the kernel
    if( mKernelSize > MAX_KERNEL_SIZE ) {
	std::cerr << "The size of the filter kernel is too large: "
		  << mKernelSize
		  << ": "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    // Copy the kernel
    checkCudaErrors( cudaMemcpyToSymbol( d_cKernel, h_mKernel, mKernelSize * mKernelSize * sizeof(T) ) );

    //
    // Set the execution configuration
    //

    // Set the tile size
    const unsigned int tileWidth = IMAGE_FILTER_BLOCK_W + 2 * mHKernelSize;
    const unsigned int tileHeight = IMAGE_FILTER_BLOCK_H + 2 * mHKernelSize;

    // Check the number of threads
    // MAX_NO_THREADS is defined in exec_config.h
    if( tileWidth * IMAGE_FILTER_THREAD_BLOCK_H > MAX_NO_THREAD ) {
	std::cerr << "Too many threads in a sub-block: "
		  << tileWidth * IMAGE_FILTER_THREAD_BLOCK_H
		  << ": "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    //
    // Check the size of the shared memory to be used
    //
    if( tileWidth * tileHeight > IMAGE_FILTER_MAX_SHARED_MEMORY_SIZE ) {
	std::cerr << "The size of the tile is too large to copy the pixel data into shared memory: "
		  << "( " << tileWidth << ", " << tileHeight << " ):"
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    // Set the execution configuration
    const unsigned int originalImageWidth = d_iImage.getImageWidth() - 2 * mHKernelSize; // removing the padding region
    const unsigned int originalImageHeight = d_iImage.getImageHeight() - 2 * mHKernelSize;
    const dim3 grid( iDivUp( originalImageWidth, IMAGE_FILTER_BLOCK_W ), iDivUp( originalImageHeight, IMAGE_FILTER_BLOCK_H ) );
    const dim3 threadBlock( tileWidth, IMAGE_FILTER_THREAD_BLOCK_H );

    //
    // Apply the filter using a GPU
    //
    checkCudaErrors( cudaDeviceSynchronize() ); // wait for threads

    if( noChannels == 3 )
	for( unsigned int i = 0; i < noChannels; i++ )
	    applyFilterGPUKernel<<<grid,threadBlock>>>( d_iImage.getImagePtr( i ), d_iImage.getImageWidth(), d_iImage.getImageHeight(),
									     IMAGE_FILTER_BLOCK_W, IMAGE_FILTER_BLOCK_H, mHKernelSize,
									     d_oImage->getImagePtr( i ), d_oImage->getImageWidth(), d_oImage->getImageHeight() );
    else
	applyFilterGPUKernel<<<grid,threadBlock>>>( d_iImage.getYCompPtr(), d_iImage.getImageWidth(), d_iImage.getImageHeight(),
						    IMAGE_FILTER_BLOCK_W, IMAGE_FILTER_BLOCK_H, mHKernelSize,
						    d_oImage->getYCompPtr(), d_oImage->getImageWidth(), d_oImage->getImageHeight() );
    
    getLastCudaError( "applyFilterGPUKernel() execution failed." );

    checkCudaErrors( cudaDeviceSynchronize() );	// wait for threads 

    return 0;

}

////
//// Explicit instantiation for the template class
////
template class imageFilter<float>;
template class imageFilter<double>;
