#ifndef COLOR_CONVERTER_H
#define COLOR_CONVERTER_H

//
// The include file for forward declaration
//
#include "image_rw_cuda_fwd.h"

//
// The color converter class
//
template <typename T>
class colorConverter {

  private:
    // You can define color conversion matrices here.

  public:
    int calYComponentGPU( dsaImage<T> *d_image );
    int calYComponentCPU( hsaImage<T> *h_image );

};

#endif // COLOR_CONVERTER_H
