# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 Jason Turner
# SPDX-FileCopyrightText: 2023 NeoN authors

# cmake-format: off
##############################################################################
# This function will enable sanitizers                                       #
# from here                                                                  #
# https://github.com/cpp-best-practices/cmake_template                       #
##############################################################################
# cmake-format: on

function(
  enable_sanitizers
  project_name
  ENABLE_SANITIZE_ADDRESS
  ENABLE_SANITIZE_LEAK
  ENABLE_SANITIZE_UNDEFINED_BEHAVIOR
  ENABLE_SANITIZE_THREAD
  ENABLE_SANITIZE_MEMORY)

  # message(FATAL_ERROR "add addr ${ENABLE_NeoN_SANITIZE_ADDRESS}")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(SANITIZERS "")

    if(${ENABLE_SANITIZE_ADDRESS})
      list(APPEND SANITIZERS "address")
    endif()

    if(${ENABLE_SANITIZE_LEAK})
      message(FATAL_ERROR "add leak")
      list(APPEND SANITIZERS "leak")
    endif()

    if(${ENABLE_SANITIZE_UNDEFINED_BEHAVIOR})
      list(APPEND SANITIZERS "undefined")
    endif()

    if(${ENABLE_SANITIZE_THREAD})
      if("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS)
        message(WARNING "Thread sanitizer does not work with Address and Leak sanitizer enabled")
      else()
        list(APPEND SANITIZERS "thread")
      endif()
    endif()

    if(${ENABLE_SANITIZE_MEMORY} AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      message(
        WARNING "Memory sanitizer requires all the code (including libc++) to be MSan-instrumented\
          otherwise it reports false positives")
      if("address" IN_LIST SANITIZERS
         OR "thread" IN_LIST SANITIZERS
         OR "leak" IN_LIST SANITIZERS)
        # message(FATAL_ERROR "Memory sanitizer does not work with Address, Thread or Leak sanitizer
        # enabled")
      else()
        list(APPEND SANITIZERS "memory")
      endif()
    endif()
  elseif(MSVC)
    if(${ENABLE_SANITIZE_ADDRESS})
      list(APPEND SANITIZERS "address")
    endif()
    if(${ENABLE_SANITIZE_LEAK}
       OR ${ENABLE_SANITIZE_UNDEFINED_BEHAVIOR}
       OR ${ENABLE_SANITIZE_THREAD}
       OR ${ENABLE_SANITIZE_MEMORY})
      message(WARNING "MSVC only supports address sanitizer")
    endif()
  endif()

  list(JOIN SANITIZERS "," LIST_OF_SANITIZERS)

  if(LIST_OF_SANITIZERS)
    if(NOT "${LIST_OF_SANITIZERS}" STREQUAL "")
      if(NOT MSVC)
        # message(FATAL_ERROR "${project_name} ${LIST_OF_SANITIZERS}")
        target_compile_options(${project_name} PRIVATE -fsanitize=${LIST_OF_SANITIZERS})
        target_link_options(${project_name} PRIVATE -fsanitize=${LIST_OF_SANITIZERS})
      else()
        string(FIND "$ENV{PATH}" "$ENV{VSINSTALLDIR}" index_of_vs_install_dir)
        if("${index_of_vs_install_dir}" STREQUAL "-1")
          message(
            SEND_ERROR
              "Using MSVC sanitizers requires setting the MSVC environment before building the \
              project. Please manually open the MSVC command prompt and rebuild the project.")
        endif()
        target_compile_options(${project_name} INTERFACE /fsanitize=${LIST_OF_SANITIZERS} /Zi
                                                         /INCREMENTAL:NO)
        target_compile_definitions(${project_name} INTERFACE _DISABLE_VECTOR_ANNOTATION
                                                             _DISABLE_STRING_ANNOTATION)
        target_link_options(${project_name} INTERFACE /INCREMENTAL:NO)
      endif()
    endif()
  endif()

endfunction()
