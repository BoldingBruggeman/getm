! Copyright (C) 2020 Bolding & Bruggeman

MODULE grid_module
!> Description:
!> This module contains basic definitions for 1D, 2D and 3D grids.
!>
!> The type_base_grid is extended to 1D, 2D and 3D grids by adding
!> the rank of the grid as well as variables to hold the grid extent
!> e.g. imin and imax. For use in models using domain decompostion
!> a vector - halo - (of lenght equal to the rank) is provided.
!> The extent of the grid (including halos) are stored in two arrays
!> l and u (lower and upper).
!> Creation of af grid can be done via several calls to the type bound
!> procedure configure - parsing optional parameters for each of the
!> variables imin, imax, .... and halo.
!> Check of a valid grid can be done through the boolean variable
!> grid_ready. If grid_ready is .false. some element have not been
!> initialized.
!> The type_grid provides a type bound procedure that prints the
!> grid content to the provided Fortran unit.
!> [Xgcm]()
!> [comodo](https://web.archive.org/web/20160417032300/http://pycomodo.forge.imag.fr/norm.html)
!>  Absolute dynamic topography
!> http://pangeo.io/use_cases/physical-oceanography/sea-surface-height.html#initialize-dataset
!> [binder](https://mybinder.org/)
!> [black](https://github.com/psf/black)
!>
!>
!> Current Code Owner: Karsten Bolding
!>
!> Language: Fortran 90.

   USE, INTRINSIC :: ISO_FORTRAN_ENV

   IMPLICIT NONE

   PRIVATE  ! Private scope by default

!  Module constants

!  Module variables
   TYPE type_base_grid
      integer :: grid_rank
        !! rank of the grid
      integer, dimension(1) :: grid_shape
      logical :: grid_ready=.false.
        !! is the grid fully configured?
      logical, private :: have_halo=.false.
        !! is the halo different from 0
!   contains
!      procedure, PASS :: create
!      procedure, PASS :: print
   END TYPE type_base_grid

   TYPE, extends(type_base_grid) :: type_1d_grid
     !! 1D grid extends type_base_grid to a 1D grid
      integer :: imin = -1
        !! minimum - 1. dimension
      integer :: imax = -1
        !! maximum - 1. dimension
      integer, dimension(1) :: halo = 0
        !! halo size
      integer, dimension(1) :: l = -1
        !! lower extent - including halo
      integer, dimension(1) :: u = -1
        !! upper extent - including halo
      logical, private :: have_imin=.false., have_imax=.false.
        !! is extent initialized
   contains
      procedure :: create => create_1d_grid
      procedure :: print => print_1d_grid
   END TYPE type_1d_grid

   TYPE, extends(type_base_grid) :: type_2d_grid
     !! 2D grid extends type_base_grid to a 2D grid
      integer :: imin = -1
        !! minimum - 1. dimension
      integer :: imax = -1
        !! maximum - 1. dimension
      integer :: jmin = -1
        !! minimum - 2. dimension
      integer :: jmax = -1
        !! maximum - 2. dimension
      integer, dimension(2) :: halo = 0
        !! halo size
      integer, dimension(2) :: l = -1
        !! lower extent - including halo
      integer, dimension(2) :: u = -1
        !! upper extent - including halo
      logical, private :: have_imin=.false., have_imax=.false.
      logical, private :: have_jmin=.false., have_jmax=.false.
        !! are all extents initialized
   contains
      procedure :: create => create_2d_grid
      procedure :: print => print_2d_grid
   END TYPE type_2d_grid

   TYPE, extends(type_base_grid) :: type_3d_grid
     !! 3D grid extends type_base_gridd to a 3D grid
      integer :: imin = -1
        !! minimum - 1. dimension
      integer :: imax = -1
        !! maximum - 1. dimension
      integer :: jmin = -1
        !! minimum - 2. dimension
      integer :: jmax = -1
        !! maximum - 2. dimension
      integer :: kmin = -1
        !! minimum - 3. dimension
      integer :: kmax = -1
        !! maximum - 3. dimension
      integer, dimension(3) :: halo = 0
        !! halo size
      integer, dimension(3) :: l = -1
        !! lower extent - including halo
      integer, dimension(3) :: u = -1
        !! upper extent - including halo
      logical, private :: have_imin=.false., have_imax=.false.
      logical, private :: have_jmin=.false., have_jmax=.false.
      logical, private :: have_kmin=.false., have_kmax=.false.
        !! are all extents initialized
   contains
      procedure :: create => create_3d_grid
      procedure :: print => print_3d_grid
   END TYPE type_3d_grid

!  Public members
   public type_1d_grid, type_2d_grid, type_3d_grid

CONTAINS

SUBROUTINE create_1d_grid(self,imin,imax,halo)

   IMPLICIT NONE

!  Subroutine arguments
   class(type_1d_grid), intent(inout) :: self
   integer, intent(in), optional :: imin,imax
   integer, intent(in), optional :: halo
      !! Grid dimensions

!  Local constants

!  Local variables
!-----------------------------------------------------------------------
   self%grid_rank = 1
   if (present(imin)) then
      self%have_imin = .true.
      self%imin = imin
   end if
   if (present(imax)) then
      self%have_imax = .true.
      self%imax = imax
   end if
   if (present(halo)) then
      self%have_halo = .true.
      self%halo = halo
   end if
   if (self%have_imin .and. self%have_imax) then
      self%l(1) = self%imin-self%halo(1); self%u(1) = self%imax+self%halo(1)
   end if
   self%grid_ready = self%have_imin .and. self%have_imax
   if (self%grid_ready .and. .not. self%have_halo) self%halo=0
   return
END SUBROUTINE create_1d_grid

!-----------------------------------------------------------------------

SUBROUTINE print_1d_grid(self,unit)

   IMPLICIT NONE

!  Subroutine arguments
   class(type_1d_grid), intent(in) :: self
   integer, intent(in) :: unit

!  Local constants

!  Local variables
!-----------------------------------------------------------------------
   write(unit,*) '1D grid:',self%grid_ready
   write(unit,*) 'rank=   ',self%grid_rank
   if(self%have_imin .and. self%have_imax) then
      write(unit,*) 'imin=  ',self%imin,'imax= ',self%imax
   else
      write(unit,*) 'i-dimension is missing'
   end if
   if (self%grid_ready) then
      if (self%have_halo) then
         write(unit,*) 'halo=  ',self%halo
      end if
      write(unit,*) 'l=     ',self%l
      write(unit,*) 'u=     ',self%u
   else
      write(unit,*) 'the grid is not fully configured yet'
   end if
   return
END SUBROUTINE print_1d_grid

!-----------------------------------------------------------------------

SUBROUTINE create_2d_grid(self,imin,imax,jmin,jmax,halo)

   IMPLICIT NONE

!  Subroutine arguments
   class(type_2d_grid), intent(inout) :: self
   integer, intent(in), optional :: imin,imax
   integer, intent(in), optional :: jmin,jmax
   integer, intent(in), dimension(2), optional :: halo
      !! Grid dimensions

!  Local constants

!  Local variables
!-----------------------------------------------------------------------
   self%grid_rank = 2
   if (present(imin)) then
      self%have_imin = .true.
      self%imin = imin
   end if
   if (present(imax)) then
      self%have_imax = .true.
      self%imax = imax
   end if
   if (present(jmin)) then
      self%have_jmin = .true.
      self%jmin = jmin
   end if
   if (present(jmax)) then
      self%have_jmax = .true.
      self%jmax = jmax
   end if
   if (present(halo)) then
      self%have_halo = .true.
      self%halo = halo
   end if
   if (self%have_imin .and. self%have_imax) then
      self%l(1) = self%imin-self%halo(1); self%u(1) = self%imax+self%halo(1)
   end if
   if (self%have_jmin .and. self%have_jmax) then
      self%l(2) = self%jmin-self%halo(2); self%u(2) = self%jmax+self%halo(2)
   end if
   self%grid_ready = self%have_imin .and. self%have_imax .and. &
                     self%have_jmin .and. self%have_jmax
   if (self%grid_ready .and. .not. self%have_halo) self%halo=0
   return
END SUBROUTINE create_2d_grid

!-----------------------------------------------------------------------

SUBROUTINE print_2d_grid(self,unit)

   IMPLICIT NONE

!  Subroutine arguments
   class(type_2d_grid), intent(in) :: self
   integer, intent(in) :: unit

!  Local constants

!  Local variables
!-----------------------------------------------------------------------
   write(unit,*) '2D grid:',self%grid_ready
   write(unit,*) 'rank=   ',self%grid_rank
   if(self%have_imin .and. self%have_imax) then
      write(unit,*) 'imin=  ',self%imin,'imax= ',self%imax
   else
      write(unit,*) 'i-dimension is missing'
      write(*,*) self%have_imin,self%have_imax
      write(*,*) self%imin,self%imax
   end if
   if(self%have_jmin .and. self%have_jmax) then
      write(unit,*) 'jmin=  ',self%jmin,'jmax= ',self%jmax
   else
      write(unit,*) 'j-dimension is missing'
   end if
   if (self%have_halo) then
      write(unit,*) 'halo=  ',self%halo
   end if
   if (self%grid_ready) then
      write(unit,*) 'l=     ',self%l
      write(unit,*) 'u=     ',self%u
   else
      write(unit,*) 'WARNING: the grid is not fully configured yet'
   end if
   return
END SUBROUTINE print_2d_grid

!-----------------------------------------------------------------------

SUBROUTINE create_3d_grid(self,imin,imax,jmin,jmax,kmin,kmax,halo)

   IMPLICIT NONE

!  Subroutine arguments
   class(type_3d_grid), intent(inout) :: self
   integer, intent(in), optional :: imin,imax
   integer, intent(in), optional :: jmin,jmax
   integer, intent(in), optional :: kmin,kmax
   integer, intent(in), dimension(3), optional :: halo
      !! Grid dimensions

!  Local constants

!  Local variables
!-----------------------------------------------------------------------
   self%grid_rank = 3
   if (present(imin)) then
      self%have_imin = .true.
      self%imin = imin
   end if
   if (present(imax)) then
      self%have_imax = .true.
      self%imax = imax
   end if
   if (present(jmin)) then
      self%have_jmin = .true.
      self%jmin = jmin
   end if
   if (present(jmax)) then
      self%have_jmax = .true.
      self%jmax = jmax
   end if
   if (present(kmin)) then
      self%have_kmin = .true.
      self%kmin = kmin
   end if
   if (present(kmax)) then
      self%have_kmax = .true.
      self%kmax = kmax
   end if
   if (present(halo)) then
      self%have_halo = .true.
      self%halo = halo
   end if
   if (self%have_imin .and. self%have_imax) then
      self%l(1) = self%imin-self%halo(1); self%u(1) = self%imax+self%halo(1)
   end if
   if (self%have_jmin .and. self%have_jmax) then
      self%l(2) = self%jmin-self%halo(2); self%u(2) = self%jmax+self%halo(2)
   end if
   if (self%have_kmin .and. self%have_kmax) then
      self%l(3) = self%kmin-self%halo(3); self%u(3) = self%kmax+self%halo(3)
   end if
   self%grid_ready = self%have_imin .and. self%have_imax .and. &
                     self%have_jmin .and. self%have_jmax .and. &
                     self%have_kmin .and. self%have_kmax
   if (self%grid_ready .and. .not. self%have_halo) self%halo=0
   return
END SUBROUTINE create_3d_grid

!-----------------------------------------------------------------------

SUBROUTINE print_3d_grid(self,unit)

   IMPLICIT NONE

!  Subroutine arguments
   class(type_3d_grid), intent(in) :: self
   integer, intent(in) :: unit

!  Local constants

!  Local variables
!-----------------------------------------------------------------------
   write(unit,*) '3D grid:',self%grid_ready
   write(unit,*) 'rank=   ',self%grid_rank
   if(self%have_imin .and. self%have_imax) then
      write(unit,*) 'imin=  ',self%imin,'imax= ',self%imax
   else
      write(unit,*) 'i-dimension is missing'
   end if
   if(self%have_jmin .and. self%have_jmax) then
      write(unit,*) 'jmin=  ',self%jmin,'jmax= ',self%jmax
   else
      write(unit,*) 'j-dimension is missing'
   end if
   if(self%have_kmin .and. self%have_kmax) then
      write(unit,*) 'kmin=  ',self%kmin,'kmax= ',self%kmax
   else
      write(unit,*) 'k-dimension is missing'
   end if
   if (self%have_halo) then
      write(unit,*) 'halo=  ',self%halo
   end if
   if (self%grid_ready) then
      write(unit,*) 'l=     ',self%l
      write(unit,*) 'u=     ',self%u
   else
      write(unit,*) 'WARNING: the grid is not fully configured yet'
   end if
   return
END SUBROUTINE print_3d_grid

!-----------------------------------------------------------------------

END MODULE grid_module
