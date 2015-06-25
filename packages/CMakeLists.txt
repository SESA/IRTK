SUBDIRS(transformation segmentation registration registration2 applications)

IF (BUILD_CARDIAC)
  SUBDIRS(cardiac)
ENDIF (BUILD_CARDIAC)


