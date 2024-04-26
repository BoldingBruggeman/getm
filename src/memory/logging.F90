! Copyright (C) 2020 Bolding & Bruggeman

MODULE logging

   !! Description:
   !!   < Say what this module contains >
   !!
   !! Current Code Owner: < Name of person responsible for this code >
   !!
   !! Code Description:
   !!   Language: Fortran 90.
   !!   This code is written to JULES coding standards v1.

   IMPLICIT NONE

   PRIVATE  ! Private scope by default

!  Module constants
   integer, parameter :: stdout = 6
   integer, parameter :: stderr = 0
   integer, parameter :: error = stderr
   integer, parameter :: debug = stderr
   integer, parameter :: indent = 2

!  Module types and variables
   type, public :: type_logging
      !! author: Karsten Bolding
      !! version: v0.1
      !!
      !! Logging type

      logical :: global_info_silence = .false.
      character(len=256), public :: prepend = ''
      integer, private :: nlog=0, nwarn=0, nerr=0, ndebug=0, ncostum=0

      contains

      procedure :: configure => logging_configure
      procedure :: info => logging_info
      procedure :: warn => logging_warn
      procedure :: error => logging_error
      procedure :: debug => logging_debug
      procedure :: costum => logging_costum

   end type type_logging

!   TYPE(type_logging), public :: logs

CONTAINS

!---------------------------------------------------------------------------

SUBROUTINE logging_configure(self)

   !! Initialize the logging module

   IMPLICIT NONE

!  Subroutine arguments
   CLASS(type_logging) :: self

!  Local constants

!  Local variables
   integer :: n
!---------------------------------------------------------------------------
!   if (present(self%prepend)) then
!      self%prepend = trim(prepend)
!   end if
   return
END SUBROUTINE logging_configure

!---------------------------------------------------------------------------

SUBROUTINE logging_info(self, msg, level, msg2, silent)

   !! Logging general information

   IMPLICIT NONE

!  Subroutine arguments
   CLASS(type_logging) :: self
   character(len = *), intent(in) :: msg
   integer, intent(in), optional :: level
   character(len = *), intent(in), optional :: msg2
   logical, intent(in), optional :: silent

!  Local constants

!  Local variables
   integer :: n
!---------------------------------------------------------------------------
   self%nlog = self%nlog+1
!KB   write(20,*) self%global_info_silence
   if (self%global_info_silence) return
   if (present(silent) .and. silent) then
      return
   end if
   if (present(level)) then
      n = level
   else
      n = 0
   end if
   if (len(self%prepend) .gt. 0) then
      write(stdout,'(a)',advance='NO') trim(self%prepend)
   end if
   write(stdout,'(i05)',advance='NO') self%nlog
   write(stdout,'(a)',advance='NO') repeat(' ',n*indent+1)
   if (present(msg2)) then
      write(stdout,'(a)', advance='NO') msg
      write(stdout,'(a)') trim(msg2)
   else
      write(stdout,'(a)') trim(msg)
   end if

   return
END SUBROUTINE logging_info

!---------------------------------------------------------------------------

SUBROUTINE logging_warn(self, msg, level)

   !! Logging warnings

   IMPLICIT NONE

!  Subroutine arguments
   CLASS(type_logging) :: self
   character(len = *) :: msg
   integer, intent(in), optional :: level

!  Local constants

!  Local variables
   integer :: n
!---------------------------------------------------------------------------

   self%nwarn = self%nwarn+1
   if (present(level)) then
      n = level
   else
      n = 0
   end if
   write(stderr,'(i05)',advance='NO') self%nwarn
   write(stderr,'(a)',advance='NO') ' WARNING '
   write(stderr,'(a)',advance='NO') repeat(' ',n*indent+1)
   write(stderr,'(a)') msg

   return
END SUBROUTINE logging_warn

!---------------------------------------------------------------------------

SUBROUTINE logging_error(self, msg, level, fatal)

   !! Logging errors - optionally fatal

   IMPLICIT NONE

!  Subroutine arguments
   CLASS(type_logging) :: self
   character(len = *) :: msg
   integer, intent(in), optional :: level
   logical, intent(in), optional :: fatal

!  Local constants

!  Local variables
   integer :: n
!---------------------------------------------------------------------------

   self%nerr = self%nerr+1
   if (present(level)) then
      n = level
   else
      n = 0
   end if
   write(stderr,'(i05)',advance='NO') self%nerr
   if(present(fatal)) then
      write(stderr,'(a)',advance='NO') ' FATAL ERROR '
   else
      write(stderr,'(a)',advance='NO') ' ERROR '
   end if
   write(stderr,'(a)',advance='NO') repeat(' ',n*indent+1)
   write(stderr,*) msg

   if(present(fatal)) then
      stop 'logging_error()'
   end if
   call flush(stderr)

   return
END SUBROUTINE logging_error

!---------------------------------------------------------------------------

SUBROUTINE logging_debug(self, msg, level)

   !! Debug information

   IMPLICIT NONE

!  Subroutine arguments
   CLASS(type_logging) :: self
   character(len = *) :: msg
   integer, intent(in), optional :: level

!  Local constants

!  Local variables
   integer :: n
!---------------------------------------------------------------------------
#ifdef _DEBUG_
   self%ndebug = self%ndebug+1
   if (present(level)) then
      n = level
   else
      n = 0
   end if
   write(debug,'(i05)',advance='NO') self%ndebug
   write(debug,'(a)',advance='NO') ' DEBUG '
   write(debug,'(a)',advance='NO') repeat(' ',n*indent+1)
   write(debug,*) msg
#endif
   return
END SUBROUTINE logging_debug

!---------------------------------------------------------------------------

SUBROUTINE logging_costum(self, unit, msg, level, msg2)

   !! Logging costum information - using unit for output

   IMPLICIT NONE

!  Subroutine arguments
   CLASS(type_logging) :: self
   integer, intent(in) :: unit
   character(len = *), intent(in) :: msg
   integer, intent(in), optional :: level
   character(len = *), intent(in), optional :: msg2

!  Local constants

!  Local variables
   integer :: n
!---------------------------------------------------------------------------

   self%ncostum = self%ncostum+1
   if (present(level)) then
      n = level
   else
      n = 0
   end if
   write(unit,'(i05)',advance='NO') self%ncostum
   write(unit,'(a)',advance='NO') repeat(' ',n*indent+1)
   if (present(msg2)) then
      write(unit,'(a)', advance='NO') msg
      write(unit,'(a)') trim(msg2)
   else
      write(unit,'(a)') trim(msg)
   end if

   return
END SUBROUTINE logging_costum

!---------------------------------------------------------------------------

END MODULE logging
