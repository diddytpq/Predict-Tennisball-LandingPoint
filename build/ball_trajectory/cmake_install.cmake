# Install script for directory: /home/yoseph/ros/Predict-Tennisball-LandingPoint/src/ball_trajectory

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/yoseph/ros/Predict-Tennisball-LandingPoint/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/yoseph/ros/Predict-Tennisball-LandingPoint/build/ball_trajectory/catkin_generated/installspace/ball_trajectory.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ball_trajectory/cmake" TYPE FILE FILES
    "/home/yoseph/ros/Predict-Tennisball-LandingPoint/build/ball_trajectory/catkin_generated/installspace/ball_trajectoryConfig.cmake"
    "/home/yoseph/ros/Predict-Tennisball-LandingPoint/build/ball_trajectory/catkin_generated/installspace/ball_trajectoryConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ball_trajectory" TYPE FILE FILES "/home/yoseph/ros/Predict-Tennisball-LandingPoint/src/ball_trajectory/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ball_trajectory" TYPE DIRECTORY FILES "/home/yoseph/ros/Predict-Tennisball-LandingPoint/src/ball_trajectory/include/ball_trajectory/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ball_trajectory" TYPE DIRECTORY FILES
    "/home/yoseph/ros/Predict-Tennisball-LandingPoint/src/ball_trajectory/launch"
    "/home/yoseph/ros/Predict-Tennisball-LandingPoint/src/ball_trajectory/models"
    "/home/yoseph/ros/Predict-Tennisball-LandingPoint/src/ball_trajectory/rviz"
    "/home/yoseph/ros/Predict-Tennisball-LandingPoint/src/ball_trajectory/worlds"
    )
endif()

