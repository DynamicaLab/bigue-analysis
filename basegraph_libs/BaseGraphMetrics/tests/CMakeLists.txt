if (DEFINED ENV{GITHUB_ACTIONS})
    include(FetchContent)
    FetchContent_Declare(
      googletest
      # Specify the commit you depend on and update it regularly.
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Manually add subdirectory to avoid installing gtest with BaseGraph library (EXCLUDE_FROM_ALL option)
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
      FetchContent_Populate(googletest)
      add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
else ()
    find_package(GTest REQUIRED)
endif ()

set(TEST_FILES_NAMES test_directed test_undirected)

foreach(TEST_NAME ${TEST_FILES_NAMES})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${TEST_NAME} ${EXTENSION_NAME} BaseGraph::core gtest gtest_main)
    add_test(${TEST_NAME} ${TEST_NAME})
endforeach()
