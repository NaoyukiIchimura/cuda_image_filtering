////
//// color_converter.cpp: the member functions for the colorConverter class
////

///
/// The standard include file
///
#include <iostream>

///
/// The include file for the color converter
///
#include "image_rw_cuda.h"
#include "color_converter.h"

////
//// The member function for the color converter class
////

///
/// Calculate Y component
///
template <typename T>
int colorConverter<T>::calYComponentCPU( hsaImage<T> *h_image )
{

    //
    // Set the pointer of the R, G, B and Y component images
    // Note that the Y component is in the first channel of the converted image
    //
    const T *rImage = h_image->getImagePtr( 0 );
    const T *gImage = h_image->getImagePtr( 1 );
    const T *bImage = h_image->getImagePtr( 2 );
    T *YImage = h_image->getYCompPtr();
    
    //
    // Calculate Y component
    //
    for( int i = 0; i < h_image->getImageHeight(); i++ ) {
	for( int j = 0; j < h_image->getImageWidth(); j++ ) {
	    int pixelPos = i * h_image->getImageWidth() + j;
	    YImage[pixelPos] = 0.2126 * rImage[pixelPos] + 0.7152 * gImage[pixelPos] + 0.0722 * bImage[pixelPos];
	}
    }

    return 0;

}

////
//// Explicit instantiation of the template class
//// The member functions are also instantiated by explicit instantiation.
//// The type of the template can be predefined.
////
template class colorConverter<float>;
template class colorConverter<double>;
