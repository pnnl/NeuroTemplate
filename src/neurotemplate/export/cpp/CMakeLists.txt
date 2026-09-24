# Build the C++ runner for an exported NeuroTemplate package. From the
# package root:
#
#     cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
#     cmake --build build
#     build/validate          # reproduce */reference/ from C++
#     build/example
#
# Dependencies: onnxruntime (shared library) and nlohmann/json (header-only).
# Both are downloaded at configure time unless found locally:
#   -DONNXRUNTIME_ROOT=<dir>  an unpacked onnxruntime release (include/, lib/)
#   nlohmann_json             any installed 3.11+ (find_package) is used first

cmake_minimum_required(VERSION 3.24)
project(neurotemplate_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

# ── onnxruntime ──────────────────────────────────────────────────
# The version the export was verified with. Pin another one by pointing
# ONNXRUNTIME_ROOT at it.
set(ONNXRUNTIME_VERSION 1.23.2)
set(ONNXRUNTIME_ROOT "" CACHE PATH "An unpacked onnxruntime release; downloaded if empty.")

if(NOT ONNXRUNTIME_ROOT)
  if(APPLE)
    set(ort_platform osx-universal2)
    set(ort_hash 49ae8e3a66ccb18d98ad3fe7f5906b6d7887df8a5edd40f49eb2b14e20885809)
  elseif(WIN32)
    set(ort_platform win-x64)
    set(ort_hash 0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ort_platform linux-aarch64)
    set(ort_hash 7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c)
  else()
    set(ort_platform linux-x64)
    set(ort_hash 1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b)
  endif()
  set(ort_archive onnxruntime-${ort_platform}-${ONNXRUNTIME_VERSION})
  if(WIN32)
    set(ort_archive ${ort_archive}.zip)
  else()
    set(ort_archive ${ort_archive}.tgz)
  endif()
  FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ort_archive}
    URL_HASH SHA256=${ort_hash})
  FetchContent_MakeAvailable(onnxruntime)
  set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
endif()

# Not find_package(onnxruntime): the release's own CMake config points its
# include directory at include/onnxruntime, which the archive does not have.
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib
             NO_DEFAULT_PATH NO_CACHE REQUIRED)

# ── nlohmann/json ────────────────────────────────────────────────
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  FIND_PACKAGE_ARGS 3.11)
FetchContent_MakeAvailable(nlohmann_json)

# ── the runtime and its two programs ─────────────────────────────
add_library(nt_runtime INTERFACE)
target_include_directories(nt_runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(nt_runtime INTERFACE ${ONNXRUNTIME_LIBRARY} nlohmann_json::nlohmann_json)

foreach(program validate example)
  add_executable(${program} ${program}.cpp)
  target_link_libraries(${program} PRIVATE nt_runtime)
  set_target_properties(${program} PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)
  if(WIN32)
    # Windows has no rpath: the DLLs must sit next to the executable.
    file(GLOB ort_dlls ${ONNXRUNTIME_ROOT}/lib/*.dll)
    add_custom_command(TARGET ${program} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${ort_dlls} $<TARGET_FILE_DIR:${program}>)
  endif()
endforeach()
