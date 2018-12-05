#ifndef IMAGE_FILTER_H
#define IMAGE_FILTER_H

//
// The include file for forward declaration
//
#include "image_rw_cuda_fwd.h"

//
// A image filter class
//
template <typename T>
class imageFilter {

  private:
    unsigned int mKernelSize;	// The size of a kernel 
    int mHKernelSize;		// The half of the size of a kernel; note the type of the variable

    T *h_mKernel;		// A filter kernel
    
  public:
    //
    // The default constructor and destructor
    //
    imageFilter();
    ~imageFilter();

    //
    // The constructor with the filter parameters 
    //
    imageFilter( const unsigned int &kernelSize, const T *h_kernel );

    //
    // Perform filtering
    //
    int applyFilterCPU( hsaImage<T> &h_iImage, const unsigned int &noChannels, hsaImage<T> *h_oImage ) const;
    int applyFilterGPU( dsaImage<T> &d_iImage, const unsigned int &noChannels, dsaImage<T> *d_oImage ) const;

};

#endif // IMAGE_FILTER_H

