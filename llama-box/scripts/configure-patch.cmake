set(VENDOR_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp)

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
    message(STATUS "Patching vendor ggml")
    file(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml/*.patch")
    foreach (PATCH_FILE ${PATCHES})
        set(PATCH_BASE "")
        foreach (CANDIDATE_BASE ${VENDOR_PATH}/ggml ${VENDOR_PATH})
            execute_process(
                    COMMAND ${GIT_EXECUTABLE} -C ${CANDIDATE_BASE} apply --check ${PATCH_FILE}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE CHECK_RESULT
                    OUTPUT_QUIET
                    ERROR_QUIET
            )
            if (CHECK_RESULT EQUAL 0)
                set(PATCH_BASE ${CANDIDATE_BASE})
                break()
            endif ()
        endforeach ()

        if (PATCH_BASE)
            execute_process(
                    COMMAND ${GIT_EXECUTABLE} -C ${PATCH_BASE} apply --whitespace=nowarn ${PATCH_FILE}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE PATCH_RESULT
            )
            if (PATCH_RESULT EQUAL 0)
                message(STATUS "  Applied ${PATCH_FILE}")
            else ()
                message(WARNING "  Failed to apply ${PATCH_FILE}")
            endif ()
        else ()
            message(WARNING "  Failed to apply ${PATCH_FILE}")
        endif ()
    endforeach ()
    message(STATUS "Patched vendor ${VENDOR_NAME}")
endforeach ()
