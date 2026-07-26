#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "leptonica" for configuration "Release"
set_property(TARGET leptonica APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(leptonica PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/leptonica-1.78.0.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "C:/bld/leptonica_1566144547025/_h_env/Library/lib/jpeg.lib;C:/bld/leptonica_1566144547025/_h_env/Library/lib/openjp2.lib;C:/bld/leptonica_1566144547025/_h_env/Library/lib/libpng.lib;C:/bld/leptonica_1566144547025/_h_env/Library/lib/tiff.lib;C:/bld/leptonica_1566144547025/_h_env/Library/lib/z.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/leptonica-1.78.0.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS leptonica )
list(APPEND _IMPORT_CHECK_FILES_FOR_leptonica "${_IMPORT_PREFIX}/lib/leptonica-1.78.0.lib" "${_IMPORT_PREFIX}/bin/leptonica-1.78.0.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
