set(VENDOR_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../stable-diffusion.cpp)

# Look for git
find_package(Git)
if (NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if (GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
    endif ()
endif ()
if (NOT Git_FOUND)
    message(FATAL_ERROR "Failed to apply patches: Git not found")
endif ()

# Apply patch
foreach (VENDOR_PATH ${VENDOR_PATHS})
    get_filename_component(VENDOR_NAME ${VENDOR_PATH} NAME)
    message(STATUS "Patching vendor ${VENDOR_NAME}")
    file(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/${VENDOR_NAME}/*.patch")
    foreach (PATCH_FILE ${PATCHES})
        execute_process(
                COMMAND ${GIT_EXECUTABLE} apply --check ${PATCH_FILE}
                WORKING_DIRECTORY ${VENDOR_PATH}
                RESULT_VARIABLE PATCH_RESULT
        )
        if (PATCH_RESULT EQUAL 0)
            execute_process(
                    COMMAND ${GIT_EXECUTABLE} apply --whitespace=nowarn ${PATCH_FILE}
                    WORKING_DIRECTORY ${VENDOR_PATH}
            )
            message(STATUS "  Applied ${PATCH_FILE}")
        else ()
            message(WARNING "  Failed to apply ${PATCH_FILE}")
        endif ()
    endforeach ()
    message(STATUS "Patched vendor ${VENDOR_NAME}")
endforeach ()
