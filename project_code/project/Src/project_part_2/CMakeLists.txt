set(BLOCK_NAME project_part_2)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIRS}
	${CUDA_TOOLKIT_INCLUDE}
	${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)


file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})
cuda_add_library(${BLOCK_NAME} MODULE
	pybind11_cuda_kernels.cpp
	my_cuda_kernels.cu
OPTIONS --cudart=shared -DWITH_OPENGL -DOPTIMIZATION_STEP=0x0 -lineinfo -arch sm_52 -use_fast_math -maxrregcount 104
DEBUG -Xptxas -v -g -G 
)

set_target_properties(${BLOCK_NAME} PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX ".pyd"
)

target_link_libraries(${BLOCK_NAME}  ${PYTHON_LIBRARIES})
####################################################### 
