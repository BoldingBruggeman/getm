! Copyright (C) 2020 Bolding & Bruggeman
!> The memory manager uses a 'poor mans template' method to allow for
!> easy allocation of memory for 1D, 2D, 3D arrays of integer(default),
!> real(real32) and real(real64). A total of 27 subroutines are provided
!> through one common interface - mm_s - through overloading.
!> Allocation can be done by specifying the extent of the cordinates
!> through vectors with the length of the rank (scalars for 1D arrays)
!> - l and u - for lower and upper.
!> In addition to allocating arrays an overloaded subroutine mm_print
!> allows for printing information about any of the allocated array types.
!> The 27 allocation routines + 9 print routines are generated via
!> pre-processing of the file memory_manager.F90.template.
!> Extending to higher ranks - or other data types is easily done.
!> The varaible - stat - is used to indicated errors.
!> Anything but 0 is an error.

MODULE memory_manager

   USE, INTRINSIC :: ISO_FORTRAN_ENV

   IMPLICIT NONE

   PRIVATE  ! Private scope by default

!  Module constants

!  Module types and variables

!  Public members
   public :: mm_s, mm_print

  interface mm_s
      module procedure mm_m_1d_int32
      module procedure mm_m_1d_real32
      module procedure mm_m_1d_real64
      module procedure mm_p_1d_int32
      module procedure mm_p_1d_real32
      module procedure mm_p_1d_real64
      module procedure mm_s_1d_int32
      module procedure mm_s_1d_real32
      module procedure mm_s_1d_real64
      module procedure mm_m_2d_int32
      module procedure mm_m_2d_real32
      module procedure mm_m_2d_real64
      module procedure mm_p_2d_int32
      module procedure mm_p_2d_real32
      module procedure mm_p_2d_real64
      module procedure mm_s_2d_int32
      module procedure mm_s_2d_real32
      module procedure mm_s_2d_real64
      module procedure mm_m_3d_int32
      module procedure mm_m_3d_real32
      module procedure mm_m_3d_real64
      module procedure mm_p_3d_int32
      module procedure mm_p_3d_real32
      module procedure mm_p_3d_real64
      module procedure mm_s_3d_int32
      module procedure mm_s_3d_real32
      module procedure mm_s_3d_real64
   end interface
   interface mm_print
      module procedure print_1d_int32
      module procedure print_1d_real32
      module procedure print_1d_real64
      module procedure print_2d_int32
      module procedure print_2d_real32
      module procedure print_2d_real64
      module procedure print_3d_int32
      module procedure print_3d_real32
      module procedure print_3d_real64
   end interface

CONTAINS

!-----------------------------------------------------------------------------

#define _RANK_ 1
#define _TYPE_ integer
#define _SIZE_ int32
#define _SUB_M_ mm_m_1d_int32
#define _SUB_P_ mm_p_1d_int32
#define _SUB_S_ mm_s_1d_int32
#define _PRINT_ print_1d_int32
#include "memory_manager.F90.template"

#define _TYPE_ real
#define _SIZE_ real32
#define _SUB_M_ mm_m_1d_real32
#define _SUB_P_ mm_p_1d_real32
#define _SUB_S_ mm_s_1d_real32
#define _PRINT_ print_1d_real32
#include "memory_manager.F90.template"

#define _TYPE_ real
#define _SIZE_ real64
#define _SUB_M_ mm_m_1d_real64
#define _SUB_P_ mm_p_1d_real64
#define _SUB_S_ mm_s_1d_real64
#define _PRINT_ print_1d_real64
#include "memory_manager.F90.template"
#undef _RANK_

#define _RANK_ 2
#define _TYPE_ integer
#define _SIZE_ int32
#define _SUB_M_ mm_m_2d_int32
#define _SUB_P_ mm_p_2d_int32
#define _SUB_S_ mm_s_2d_int32
#define _PRINT_ print_2d_int32
#include "memory_manager.F90.template"

#define _TYPE_ real
#define _SIZE_ real32
#define _SUB_M_ mm_m_2d_real32
#define _SUB_P_ mm_p_2d_real32
#define _SUB_S_ mm_s_2d_real32
#define _PRINT_ print_2d_real32
#include "memory_manager.F90.template"

#define _TYPE_ real
#define _SIZE_ real64
#define _SUB_M_ mm_m_2d_real64
#define _SUB_P_ mm_p_2d_real64
#define _SUB_S_ mm_s_2d_real64
#define _PRINT_ print_2d_real64
#include "memory_manager.F90.template"
#undef _RANK_

#define _RANK_ 3
#define _TYPE_ integer
#define _SIZE_ int32
#define _SUB_M_ mm_m_3d_int32
#define _SUB_P_ mm_p_3d_int32
#define _SUB_S_ mm_s_3d_int32
#define _PRINT_ print_3d_int32
#include "memory_manager.F90.template"

#define _TYPE_ real
#define _SIZE_ real32
#define _SUB_M_ mm_m_3d_real32
#define _SUB_P_ mm_p_3d_real32
#define _SUB_S_ mm_s_3d_real32
#define _PRINT_ print_3d_real32
#include "memory_manager.F90.template"

#define _TYPE_ real
#define _SIZE_ real64
#define _SUB_M_ mm_m_3d_real64
#define _SUB_P_ mm_p_3d_real64
#define _SUB_S_ mm_s_3d_real64
#define _PRINT_ print_3d_real64
#include "memory_manager.F90.template"
#undef _RANK_

!---------------------------------------------------------------------------

END MODULE memory_manager
