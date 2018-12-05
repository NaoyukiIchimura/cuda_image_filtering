#ifndef IMAGE_MSE_H
#define IMAGE_MSE_H

//
// Function prototype
//
template <typename T>
T calMSE( hsaImage<T> &h_image1, hsaImage<T> &h_image2, const unsigned int &noChannels );

#endif // IMAGE_MSE_H
