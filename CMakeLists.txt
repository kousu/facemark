#  minimum required cmake version: 3.12.0
cmake_minimum_required(VERSION 3.12.0)

project(FacemarkTrain)

option( CMAKE_BUILD_TYPE RELEASE )
option( CMAKE_INSTALL_PREFIX "../install" )


find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

#Add the source files
#add_subdirectory(src)

file(GLOB FreedomV2_SRC
    "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/*.c"
)
add_executable(FacemarkTrain ${FreedomV2_SRC})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT FacemarkTrain)
set_property(TARGET FacemarkTrain PROPERTY CXX_STANDARD 11)
target_link_libraries(FacemarkTrain ${DEPENDENCIES} ${OpenCV_LIBS})

set_target_properties (FacemarkTrain PROPERTIES
	FOLDER "FacemarkTrain"
)

install(
	TARGETS

	FacemarkTrain

	RUNTIME DESTINATION
	${CMAKE_INSTALL_PREFIX}/bin
)
