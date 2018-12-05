////
//// image_mse.cpp: calculate the mean squared error (MSE) between two images
////

///
/// The standard include files
///
#include <iostream>

#include <cstdlib>

///
/// The include file for computing MSE
///
#include "image_rw_cuda.h"

///
/// The function for calculating MSE between two images
///
template <typename T>
T calMSE( hsaImage<T> &h_image1, hsaImage<T> &h_image2, const unsigned int &noChannels )
{

    //
    // Check the number of channels
    //
    if( noChannels != 3 && noChannels != 1 ) {
	std::cerr << "The number of channels must be 3 or 1: "
		  << noChannels 
		  << ", " << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    //
    // Check the sizes of the images
    //
    if( h_image1.getImageWidth() != h_image2.getImageWidth() || h_image1.getImageHeight() != h_image2.getImageHeight() ) {
	std::cerr << "The sizes of the images for computing the MSE must be the same: "
		  << "( " << h_image1.getImageWidth() << ", " << h_image1.getImageHeight() << " )" << std::endl
		  << "( " << h_image2.getImageWidth() << ", " << h_image2.getImageHeight() << " )" << std::endl;
	exit(1);
    }

    //
    // Set the pointers of the images
    //
    T **imagePtr1;
    try {
	imagePtr1 = new T *[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the momery space for imagePtr1: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( noChannels == 3 ) // RGB image
	for( unsigned int i = 0; i < noChannels; i++ )
	    imagePtr1[i] = h_image1.getImagePtr( i );
    else if( noChannels == 1 ) // Y Component
	imagePtr1[0] = h_image1.getYCompPtr();

    T **imagePtr2;
    try {
	imagePtr2 = new T *[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the momery space for imagePtr2: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( noChannels == 3 ) // RGB image
	for( unsigned int i = 0; i < noChannels; i++ )
	    imagePtr2[i] = h_image2.getImagePtr( i );
    else if( noChannels == 1 ) // Y Component
	imagePtr2[0] = h_image2.getYCompPtr();

    //
    // Calculate the difference between the two images
    //
    T *channelMSE;
    try {
	channelMSE = new T[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for channelMSE: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    for( unsigned int i = 0; i < noChannels; i++ ) {
	unsigned int noPixel = 0;
	channelMSE[i] = 0.0;
	for( unsigned int j = 0; j < h_image1.getImageHeight(); j++ ) {
	    for( unsigned int k = 0; k < h_image1.getImageWidth(); k++ ) {
		noPixel++;
		unsigned int pixelPos = j * h_image1.getImageWidth() + k;
		channelMSE[i] += ( imagePtr1[i][pixelPos] - imagePtr2[i][pixelPos] ) * ( imagePtr1[i][pixelPos] - imagePtr2[i][pixelPos] );
	    }
	}
	channelMSE[i] /= static_cast<T>(noPixel);
    }
    
    T MSE = 0.0;
    for( unsigned int i = 0; i < noChannels; i++ )
	MSE += channelMSE[i];
    MSE /= static_cast<T>(noChannels);

    //
    // Delete the memory space
    //
    delete [] channelMSE;
    channelMSE = 0;

    delete [] imagePtr2;
    imagePtr2 = 0;
    delete [] imagePtr1;
    imagePtr1 = 0;

    return MSE;

}

////
//// Explicit instantiation of the template function
////
template
float calMSE( hsaImage<float> &h_image1, hsaImage<float> &h_image2, const unsigned int &noChannels );
template
double calMSE( hsaImage<double> &h_image1, hsaImage<double> &h_image2, const unsigned int &noChannels );
