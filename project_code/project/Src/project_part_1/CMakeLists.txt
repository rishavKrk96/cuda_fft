include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable( project_part_1
${HEADERS}
project_part_1.cu
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 
)

target_link_libraries( 
project_part_1

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

target_link_libraries( 
project_part_1
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)

