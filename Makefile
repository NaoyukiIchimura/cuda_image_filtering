#
#  Make file for cuda_image_filering
#

#
# Macros
#
IMG_LDFLAG	= -ltiff -lpng -ljpeg
LDFLAGS 	= $(IMG_LDFLAG) -lm

CUDA_INCFLAG	= -I/home/ichimura/NVIDIA_CUDA-9.2_Samples/common/inc
INCFLAGS	= $(CUDA_INCFLAG)

CC		= nvcc
CFLAGS		= -gencode arch=compute_30,code=sm_30 \
		  -gencode arch=compute_52,code=sm_52 \
		  -gencode arch=compute_61,code=sm_61 \
		  --fmad=false \
		  -O3 -std=c++11

CPP_SRCS	= cuda_image_filtering_main.cpp \
		  cuda_image_filtering_options.cpp \
		  padding.cpp \
		  color_converter.cpp \
		  image_filter.cpp \
		  image_mse.cpp \
		  postprocessing.cpp \
		  path_handler.cpp \
		  image_rw_cuda.cpp \
		  get_micro_second.cpp

CPP_HDRS	= cuda_image_filtering_options.h \
		  padding.h \
		  color_converter.h \
		  image_filter.h \
		  image_mse.h \
		  postprocessing.h \
		  path_handler.h \
		  image_rw_cuda.h \
		  image_rw_cuda_fwd.h \
		  get_micro_second.h

CU_SRCS		= color_converter_kernel.cu \
		  image_filter_kernel.cu \
		  postprocessing_kernel.cu

CU_HDRS		= color_converter_kernel.h \
		  image_filter_kernel.h \
		  postprocessing_kernel.h \
		  exec_config.h \
		  constmem_type.h

CPP_OBJS	= $(CPP_SRCS:.cpp=.o) 
CU_OBJS		= $(CU_SRCS:.cu=.o)
TARGET		= cuda_image_filtering

CPP_DEPS	= $(CPP_SRCS:.cpp=.d)
CU_DEPS		= $(CU_SRCS:.cu=.d)
DEP_FILE	= Makefile.dep

#
# Suffix rules
#
.SUFFIXES: .cpp
.cpp.o:
	$(CC) $(INCFLAGS) $(CFLAGS)  -c $<

.SUFFIXES: .cu
.cu.o:
	$(CC) $(INCFLAGS) $(CFLAGS)  -c $<

.SUFFIXES: .d
.cpp.d:
	$(CC) $(INCFLAGS) -M $< > $*.d
.cu.d:
	$(CC) $(INCFLAGS) -M $< > $*.d

#
# Generating the target
#
all: $(DEP_FILE) $(TARGET) 

#
# Linking the execution file
#
$(TARGET) : $(CU_OBJS) $(CPP_OBJS) 
	$(CC) -o $@ $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS)

#
# Generating and including dependencies
#
depend: $(DEP_FILE)
$(DEP_FILE) : $(CPP_DEPS) $(CU_DEPS)
	cat $(CPP_DEPS) $(CU_DEPS) > $(DEP_FILE)
ifeq ($(wildcard $(DEP_FILE)),$(DEP_FILE))
include $(DEP_FILE)
endif

#
# cleaning the files
#
clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) $(CPP_DEPS) $(CU_DEPS) $(DEP_FILE) $(TARGET) *~
