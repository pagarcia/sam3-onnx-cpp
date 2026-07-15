cmake_minimum_required(VERSION 3.14)
project(SAM3_ONNX)

include(CTest)

set(TARGET_NAME "Segment")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(MSVC)
  add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
  add_compile_options(/O2)
else()
  add_compile_options(-O3 -fno-omit-frame-pointer)
endif()

if(BUILD_TESTING)
  add_executable(ArtifactResolverTests
    tests/ArtifactResolverTests.cpp
  )
  target_include_directories(ArtifactResolverTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  add_test(NAME ArtifactResolverTests COMMAND ArtifactResolverTests)
endif()

if(NOT DEFINED OpenCV_DIR)
  message(FATAL_ERROR "Please set OpenCV_DIR (for example -DOpenCV_DIR=\"C:/Program Files/OpenCV/Release\")")
endif()

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set ONNXRUNTIME_DIR (for example -DONNXRUNTIME_DIR=\"C:/Program Files/onnxruntime-win-x64-gpu-1.22.1\")")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION} at: ${OpenCV_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})

if(WIN32)
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
elseif(APPLE)
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
  set(CMAKE_MACOSX_BUNDLE YES)
  set(CMAKE_BUILD_RPATH "@loader_path/../Frameworks")
  set(CMAKE_INSTALL_RPATH "@loader_path/../Frameworks;@executable_path/../Frameworks")
else()
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
  set(CMAKE_BUILD_RPATH "$ORIGIN")
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

if(WIN32)
  set(OPENFILEDIALOG_SRC src/openFileDialog.cpp)
elseif(APPLE)
  set(OPENFILEDIALOG_SRC src/openFileDialog.mm)
  set_source_files_properties(src/openFileDialog.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
  set(OPENFILEDIALOG_SRC src/openFileDialog.cpp)
endif()

add_executable(${TARGET_NAME}
  src/main.cpp
  src/onnx_test_image.cpp
  src/onnx_test_video.cpp
  src/SAM3Session.cpp
  src/SAM3Image.cpp
  src/SAM3Video.cpp
  ${OPENFILEDIALOG_SRC}
)

if(APPLE)
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
    "-framework Cocoa"
  )

  set_target_properties(${TARGET_NAME} PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/Info.plist.in"
    BUILD_RPATH "@loader_path/../Frameworks"
    INSTALL_RPATH "@loader_path/../Frameworks;@executable_path/../Frameworks;${CMAKE_INSTALL_RPATH}"
  )
elseif(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
  )

  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying ONNX Runtime DLLs..."
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ONNXRUNTIME_DIR}/lib"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
  )

  set(OPENCV_RUNTIME_BIN_DIR "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin")
  file(GLOB OPENCV_RUNTIME_DLLS "${OPENCV_RUNTIME_BIN_DIR}/*.dll")
  if(OPENCV_RUNTIME_DLLS)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Copying OpenCV DLLs..."
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${OPENCV_RUNTIME_DLLS}
              "$<TARGET_FILE_DIR:${TARGET_NAME}>"
    )
  else()
    message(WARNING "No OpenCV runtime DLLs found under ${OPENCV_RUNTIME_BIN_DIR}")
  endif()
else()
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
  )
endif()

if(APPLE)
  install(TARGETS ${TARGET_NAME}
          BUNDLE DESTINATION .
          RUNTIME DESTINATION bin)
  file(GLOB ORT_DYLIBS "${ONNXRUNTIME_DIR}/lib/libonnxruntime*.dylib")
  if(ORT_DYLIBS)
    install(FILES ${ORT_DYLIBS}
            DESTINATION "${TARGET_NAME}.app/Contents/Frameworks")
  endif()
  include(InstallRequiredSystemLibraries)
  set(CPACK_PACKAGE_NAME "SAM3_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "DragNDrop")
  include(CPack)
elseif(WIN32)
  install(TARGETS ${TARGET_NAME}
          RUNTIME DESTINATION bin)
  install(DIRECTORY "${ONNXRUNTIME_DIR}/lib/"
          DESTINATION bin
          FILES_MATCHING PATTERN "onnxruntime*.dll")
  set(CPACK_PACKAGE_NAME "SAM3_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "ZIP")
  include(CPack)
else()
  install(TARGETS ${TARGET_NAME}
          RUNTIME DESTINATION bin)
  set(CPACK_PACKAGE_NAME "SAM3_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "TGZ")
  include(CPack)
endif()
