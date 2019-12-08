cmake_minimum_required(VERSION 2.8.11)
project(FISHEYE_STEREO)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(disparity disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS})

add_executable(disparity_rectify disparity_rectify.cpp)
target_link_libraries(disparity_rectify ${OpenCV_LIBS})

add_executable(epipolar epipolar.cpp)
target_link_libraries(epipolar ${OpenCV_LIBS})

add_executable(epipolar_rectify epipolar_rectify.cpp)
target_link_libraries(epipolar_rectify ${OpenCV_LIBS})

add_executable(epipolar_right epipolar_right.cpp)
target_link_libraries(epipolar_right ${OpenCV_LIBS})

add_executable(undistort undistort.cpp)
target_link_libraries(undistort ${OpenCV_LIBS})

add_executable(undistort_rectify undistort_rectify.cpp)
target_link_libraries(undistort_rectify ${OpenCV_LIBS})

add_executable(project_points project_points.cpp)
target_link_libraries(project_points ${OpenCV_LIBS})

add_executable(distort_points distort_points.cpp)
target_link_libraries(distort_points ${OpenCV_LIBS})