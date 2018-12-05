#ifndef IMAGE_FILTER_KERNEL_H
#define IMAGE_FILTER_KERNEL_H

//
// The maximum size of the filter kernel that defines the size of the constant variable
//
const unsigned int MAX_KERNEL_SIZE = 39; // for double variables
const unsigned int MAX_HKERNEL_SIZE = MAX_KERNEL_SIZE / 2;

//
// Execution configuration
//
const unsigned int IMAGE_FILTER_BLOCK_W = 32;
const unsigned int IMAGE_FILTER_BLOCK_H = 32;
const unsigned int IMAGE_FILTER_THREAD_BLOCK_H = 8;

//
// The maximum size of the shared memory to be used
//
const unsigned int IMAGE_FILTER_MAX_SHARED_MEMORY_SIZE = ( IMAGE_FILTER_BLOCK_W + 2 * MAX_HKERNEL_SIZE ) * ( IMAGE_FILTER_BLOCK_H + 2 * MAX_HKERNEL_SIZE );

#endif // IMAGE_FILTER_KERNEL_H
