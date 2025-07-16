cmake_minimum_required(VERSION 3.8) 
project(Batch_Train_Store)               

set(CMAKE_CXX_STANDARD      17)     
set(CMAKE_C_STANDARD        11)     

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/) 
option(BUILD_SHARED_LIBS    ""  OFF) 

aux_source_directory(./                     cur_src) 

# 如果需要，可以使用aux_source_directory增加目录

add_executable(code_train                   ${cur_src}) 

# 以下可以根据需要增加需要链接的库
#if (NOT WIN32)
#    target_link_libraries(code_craft  pthread  rt  m)
#endif (NOT WIN32)
