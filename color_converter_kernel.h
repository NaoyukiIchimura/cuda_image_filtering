#ifndef COLOR_CONVERTER_KERNEL_H
#define COLOR_CONVERTER_KERNEL_H

//
// Block and thread sizes for computing Y component
//
const unsigned int Y_COMP_BLOCK_W = 32;
const unsigned int Y_COMP_BLOCK_H = 32;

const unsigned int Y_COMP_THREAD_BLOCK_W = Y_COMP_BLOCK_W;
const unsigned int Y_COMP_THREAD_BLOCK_H = Y_COMP_BLOCK_H;

#endif // COLOR_CONVERTER_KERNEL_H
