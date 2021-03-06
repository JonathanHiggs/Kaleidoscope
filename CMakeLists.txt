cmake_minimum_required(VERSION 3.16)

project(Kaleidoscope)

SET(CMAKE_GENERATOR_PLATFORM x64)
SET(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "/EHsc /wd4005 /wd4100 /wd4141 /wd4189 /wd4458 /wd4244 /wd4245 /wd4267 /wd4246")
set(CMAKE_CXX_FLAGS_DEBUG "/Od /Oi /Zi /JMC")

find_package(Clang CONFIG REQUIRED)
find_package(LLD CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)

set(FILES
    main.cpp
)

add_executable(Kaleidoscope ${FILES})

target_link_libraries(Kaleidoscope PRIVATE
    LTO
    LLVMMC
    LLVMLTO
)

target_include_directories(Kaleidoscope PRIVATE
    ${LLVM_INCLUDE_DIR}
)