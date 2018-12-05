////
//// image_filter.cpp: The member functions for the imageFilter class
////

///
/// The standard include files
///
#include <iostream>

#include <cstdlib>

///
/// The include file for an image filter
///
#include "image_filter.h"
#include "image_rw_cuda.h"

///
/// The default constructor and destructor
///
template <typename T>
imageFilter<T>::imageFilter() : mKernelSize(0), mHKernelSize(0),
				h_mKernel(0)
{

    std::cerr << "The parameters of an filter, filter size and filter kernel, are required to used the class."
	      << std::endl;
    exit(1);

}

template <typename T>
imageFilter<T>::~imageFilter()
{

    //
    // Delete the memory space for the kernel
    //
    delete [] h_mKernel;
    h_mKernel = 0;

}

///
/// The constructor with the filter parameters
///
template <typename T>
imageFilter<T>::imageFilter( const unsigned int &kernelSize, const T *h_kernel )
{

    //
    // Check if the kernel size is odd number
    //
    if( kernelSize % 2 == 0 ) {
	std::cerr << "The kernel size of a filter must be odd number: "
		  << kernelSize
		  << std::endl;
	exit(1);
    }

    //
    // Set the kernel size
    //
    mKernelSize = kernelSize;

    //
    // Set the half of the kernel size
    //
    mHKernelSize = mKernelSize / 2;

    //
    // Allocate the memory space for the filter kernel
    //
    try {
	h_mKernel = new T[ mKernelSize * mKernelSize ];
    } catch( std::bad_alloc &) {
	std::cerr << "Could not allocate the memory space for h_mKernel: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    //
    // Set the filter kernel
    // Note that the 1D array is used for CUDA
    //
    for( unsigned int i = 0; i < mKernelSize; i++ ) {
	for( unsigned int j = 0; j < mKernelSize; j++ ) {
	    unsigned int coefPos = i * mKernelSize + j;
	    h_mKernel[coefPos] = h_kernel[coefPos];
	}
    }
	    

}

///
/// Perform filter by a CPU
///
template <typename T>
int imageFilter<T>::applyFilterCPU( hsaImage<T> &h_iImage, const unsigned int &noChannels, hsaImage<T> *h_oImage ) const
{

    //
    // Allocate the memory space for the pointers of the images
    //
    T **iImagePtr;
    try {
	iImagePtr = new T *[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for iImagePtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    T **oImagePtr;
    try {
	oImagePtr = new T *[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for oImagePtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    //
    // Set the pointer of the images
    //
    if( noChannels == 3 ) { // RGB image
	for( unsigned int i = 0; i < noChannels; i++ ) {
	    iImagePtr[i] = h_iImage.getImagePtr( i );
	    oImagePtr[i] = h_oImage->getImagePtr( i );
	}
    } else if( noChannels == 1 ) { // Black and white image; Y component is used
	iImagePtr[0] = h_iImage.getYCompPtr();
	oImagePtr[0] = h_oImage->getYCompPtr();
    } else {
	std::cerr << "The number of channels must be 3 or 1: "
		  << noChannels << ", "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    //
    // Perform filtering
    //
    for( unsigned int channelNo = 0; channelNo < noChannels; channelNo++ ) {

	for( unsigned int i = mHKernelSize; i < h_iImage.getImageHeight() - mHKernelSize; i++ ) {
	    for( unsigned int j = mHKernelSize; j < h_iImage.getImageWidth() - mHKernelSize; j++ ) {

		const unsigned int oPixelPos = ( i - mHKernelSize ) * h_oImage->getImageWidth() + ( j - mHKernelSize );
		oImagePtr[channelNo][oPixelPos] = 0.0;
		for( int k = -mHKernelSize; k <= mHKernelSize; k++ ) {
		    for( int l = -mHKernelSize; l <= mHKernelSize; l++ ) {
			const unsigned int iPixelPos = ( i + k ) * h_iImage.getImageWidth() + ( j + l );
			const unsigned int coefPos = ( k + mHKernelSize ) * mKernelSize + ( l + mHKernelSize );
			oImagePtr[channelNo][oPixelPos] += iImagePtr[channelNo][iPixelPos] * h_mKernel[coefPos];
		    }
		}
		
	    }
	}	    

    }

    //
    // Delete the memory space
    //
    delete [] oImagePtr;
    oImagePtr = 0;

    delete [] iImagePtr;
    iImagePtr = 0;

    return 0;
    
}

////
//// Explicit instantiation for the template class 
////
template class imageFilter<float>;
template class imageFilter<double>;
