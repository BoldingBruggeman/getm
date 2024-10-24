! Copyright (C) 2024 Bolding & Bruggeman and Hans Burchard

module c_adaptive

   use iso_c_binding, only: c_int, c_double

   implicit none

!! @note
!! The adaptive coordinates implementation is heavily influenced by the
!! work of Bjarne Büchmann implementation - which in turn was based on
!! the work by Richard Hoffmeister.
!! https://sourceforge.net/p/getm/code/ci/iow/tree/src/3d/adaptive_coordinates_6.F90
!! and
!! https://sourceforge.net/p/getm/code/ci/iow/tree/src/3d/adaptive_coordinates.F90
!! Thes basic concept is to create a diffusion equation for the layer
!! heights where the diffusivity is build up as a sum of different
!! contributions attibuted to different constrains/flow features.
!!
!! The code by Bjarne Büchmann is well documented - this implementation
!!  will not repeat all the comments but follow the structure of the
!! original implementation.
!!
!! The vertical diffusion operator in pygetm works on a full 3D field
!! opposed to legacy GETM - and - hence the column oriented layout
!! is replaced by a field oriented approach.
!!
!! There are 3 main steps involved:
!!    1. Construct the grid diffusivity - nu (renamed from Dgrid). This will be a sequence of
!!       individual contributions
!!       1. Define background value (sigma)
!!       2. Add GVC tendency
!!       3. Add "mid-column" zooming (punish relatively big cells)
!!       4. Add surface- and bottom-cell zooming tendencies
!!       5. Add NN and SS tendencies
!!       6. Reduce if cells are already appraching "too small"
!!    2. Lateral filtering of the constructed diffusivity
!!    3. Apply the vertical filter.
!! @endnote

contains

subroutine c_update_adaptive(nx, ny, nz, halox, haloy, &
                             mask, H, D, zo, ho, &
                             NN, SS, nu, &
                             decay, hpow,  &
                             chsurf, hsurf,  &
                             chmidd, hmidd, &
                             chbott, hbott, &
                             cneigh, rneigh, &
                             cNN, drho, &
                             cSS, dvel, &
                             chmin, hmin, &
                             dt) bind(c)


!  Subroutine arguments
   integer(c_int), intent(in), value :: nx, ny, nz
   integer(c_int), intent(in), value :: halox, haloy
#define _D2_  -halox+1:nx+halox,-haloy+1:ny+haloy
#define _D3_  -halox+1:nx+halox,-haloy+1:ny+haloy,nz
   integer(c_int), intent(in) :: mask(_D2_)
   real(c_double), intent(in) :: H(_D2_)
   real(c_double), intent(in) :: D(_D2_)
   real(c_double), intent(in) :: zo(_D2_)
   real(c_double), intent(in) :: ho(_D3_)
   real(c_double), intent(in) :: NN(_D3_)
   real(c_double), intent(in) :: SS(_D3_)
   real(c_double), intent(inout) :: nu(_D3_)
#undef _D3_
#undef _D2_
   real(c_double), intent(in), value :: decay
   integer(c_int), intent(in), value :: hpow
   real(c_double), intent(in), value :: chsurf, hsurf
   real(c_double), intent(in), value :: chmidd, hmidd
   real(c_double), intent(in), value :: chbott, hbott
   real(c_double), intent(in), value :: cneigh, rneigh
   real(c_double), intent(in), value :: cNN, drho
   real(c_double), intent(in), value :: cSS, dvel
   real(c_double), intent(in), value :: chmin, hmin
   real(c_double), intent(in), value :: dt

!  Local constants
   integer, parameter ::surface=1, interior=2, bottom=3

!  Local variables
   integer, parameter :: rk=c_double
   integer :: i,j,k
   integer :: imin=1, jmin=1, imax, jmax, kmax
   real(c_double), allocatable, save :: ihmax(:,:,:)
   !KB - maybe real(c_double), allocatable, save :: havg(:,:)
   real(c_double), allocatable, save :: sdecay(:)
   real(c_double), allocatable, save :: bdecay(:)
   real(c_double), allocatable, save :: haux(:,:,:)
   real(c_double), parameter :: ceps = 1._rk/100000._rk
   logical, save :: first=.true.

!-----------------------------------------------------------------------------
   imax=nx; jmax=ny; kmax=nz

   if (first) then

#define _PRINT_
#ifdef _PRINT_
   write(100,*) 'c_update_adaptive'
   write(100,*) nx, ny, nz
   write(100,*) halox, haloy
   write(100,*) halox, haloy
!   write(100,*) H
!   write(100,*) D
!   write(100,*) ho
!   write(100,*) 'NN ', NN
!   write(100,*) 'SS ', SS
   write(100,*) 'decay ',decay, hpow
   write(100,*) 'S     ',chsurf, hsurf
   write(100,*) 'I     ',chmidd, hmidd
   write(100,*) 'B     ',chbott, hbott
   write(100,*) 'neigh ',cneigh, rneigh
   write(100,*) 'NN    ',cNN, drho
   write(100,*) 'SS    ',cSS, dvel
   write(100,*) 'min   ',hmin, chmin
#endif

      allocate(ihmax(nx,ny,3))
      allocate(sdecay(0:nz))
      allocate(bdecay(0:nz))
      allocate(haux(nx,ny,nz))

      do j=jmin,jmax
         do i=imin,imax
            if (mask(i,j) < 1 ) cycle

            ! surface
            if (chsurf > ceps) then
               if (abs(hsurf) < ceps) then
                  ihmax(i,j,surface) = 1._rk
               else if (hsurf > 0) then
                  ihmax(i,j,surface) = 1._rk/hsurf
               else
                  ihmax(i,j,surface) = kmax/(-hsurf*H(i,j))
               end if
            end if

            ! interior
            if (chmidd > ceps) then
               if (abs(hmidd) < ceps) then
                  ihmax(i,j,interior) = 1._rk
               else if (hmidd > 0) then
                  ihmax(i,j,interior) = 1._rk/hmidd
               else
                  ihmax(i,j,interior) = kmax/(-hmidd*H(i,j))
               end if
            end if

            ! bottom
            if (chbott > ceps) then
               if (abs(hbott) < ceps) then
                  ihmax(i,j,bottom) = 1._rk
               else if (hbott > 0) then
                  ihmax(i,j,bottom) = 1._rk/hbott
               else
                  ihmax(i,j,bottom) = kmax/(-hbott*H(i,j))
               end if
            end if
         end do
      end do

      if (decay > ceps) then
         ! sdecay - wcolsurf
         sdecay(0) = 0
         sdecay(kmax) = 1
         do k=kmax-1,1,-1
            sdecay(k) = sdecay(k+1)*decay
         end do

         ! bdecay - wcolbott
         bdecay(0) = 0
         bdecay(1) = 1
         do k=2,kmax
            bdecay(k) = bdecay(k-1)*decay
         end do
      end if

#ifdef _PRINT_
   write(100,*) ihmax(50,1,:)
   !write(100,*) sdecay
   !write(100,*) bdecay
   write(100,*) 'c_update_adaptive'
#endif

      first = .false.
   end if

   ! set up ratio between old and new
   do j=jmin,jmax
      do i=imin,imax
         if (mask(i,j) < 1 ) cycle

         haux(i,j,:)=D(i,j)/(H(i,j)+zo(i,j))*ho(i,j,:)
      end do
   end do
#ifdef _PRINT_
   write(100,*) 'haux ',haux(50,1,:)
#endif

   ! Construct the grid diffusivity

   ! 1.1 - sigma as background - done in Python

   ! 1.2 - GVC - done in Python

   ! 1.3 - chmidd
   large_cell_limitation: block
   real(c_double) :: relh(kmax)
   real(c_double) :: wwh

   if (chmidd > ceps) then
      do j=imin,jmax
         do i=imin,imax
            if (mask(i,j) < 1) cycle

            relh(:)=haux(i,j,:)*ihmax(i,j,interior)-0.5_rk
            do k=1,kmax
               if (relh(k) > 0._rk .and. relh(k) < 1._rk) then
                  wwh=relh(k)**hpow
               else
                  wwh=0.5_rk+sign(0.5_rk,relh(k))
               end if
               nu(i,j,k)=nu(i,j,k)+chmidd*wwh
            end do
         end do
      end do
   end if
   end block large_cell_limitation

   ! 1.4.1 chsurf
   surface_cell_limitation: block
   real(c_double) :: relh
   real(c_double) :: wwh

   if (chsurf > ceps) then
      do j=jmin,jmax
         do i=imin,imax
            if (mask(i,j) < 1) cycle

            relh=haux(i,j,kmax)*ihmax(i,j,surface)-0.5_rk
            if (relh > 0._rk .and. relh < 1._rk) then
               wwh=relh**hpow
            else
               wwh=0.5_rk+sign(0.5_rk,relh)
            end if

            nu(i,j,:)=nu(i,j,:)+chsurf*wwh*sdecay(:)
         end do
      end do
   end if
   end block surface_cell_limitation

   ! 1.4.2 chbott
   bottom_cell_limitation: block
   real(c_double) :: relh
   real(c_double) :: wwh

   if (chbott > ceps) then
      do j=jmin,jmax
         do i=imin,imax
            if (mask(i,j) < 1) cycle

            relh=haux(i,j,1)*ihmax(i,j,bottom)-0.5_rk
            if (relh > 0._rk .and. relh < 1._rk) then
               wwh=relh**hpow
            else
               wwh=0.5_rk+sign(0.5_rk,relh)
            end if

            nu(i,j,:)=nu(i,j,:)+chbott*wwh*bdecay(:)
         end do
      end do
   end if
   end block bottom_cell_limitation

   ! 1.? Neighbor-cell size-ratio limiter.
   neighbor_cell_limitation: block
   real(c_double) :: irneigh
   real(c_double) :: relh(kmax)
   real(c_double) :: wwh(kmax)
!KB   real(c_double) :: wwh

   if (cneigh > ceps) then
      irneigh = 1._rk/rneigh
      do j=jmin,jmax
         do i=imin,imax
            if (mask(i,j) < 1) cycle

            relh(1)=haux(i,j,1)/haux(i,j,2)
            relh(kmax)=haux(i,j,kmax)/haux(i,j,kmax-1)
            do k=2,kmax-1
               relh(k)=haux(i,j,k-1)/haux(i,j,k+1)
            end do
            relh(:)=max(0._rk, relh(:)-1._rk)
#if 0
            wwh(:)=min(1._rk,irneigh*relh(:))
            wwh(:) = wwh(:)**hpow
            do k=1,kmax
               nu(i,j,k)=nu(i,j,k)+cneigh*wwh(k)
            end do
#else
            nu(i,j,:)=nu(i,j,:)+cneigh*min(1._rk,irneigh*relh(:))**hpow
#endif
         end do
      end do
   end if
   end block neighbor_cell_limitation

   ! 1.5.1 bouyancy
   bouyancy: block
   real(c_double) :: idNN
   real(c_double) :: x,y

   if (cNN > ceps) then
      idNN = 1025._rk/(9.81_rk*drho)/kmax
      do j=jmin,jmax
         do i=imin,imax
            if (mask(i,j) < 1) cycle

            x = idNN*D(i,j)
            do k=1,kmax
               y = min(1._rk, x*max(0._rk, 0.5_rk*(NN(i,j,k)+NN(i,j,k-1))))
               nu(i,j,k)=nu(i,j,k)+cNN*y
            end do
         end do
      end do
   end if
   end block bouyancy

   ! 1.5.2 shear
   shear: block
   real(c_double) :: idvel
   real(c_double) :: x,y

   if (cSS > ceps) then
      idvel = 1._rk/dvel/kmax
      do j=jmin,jmax
         do i=imin,imax
            if (mask(i,j) < 1) cycle

            x = idvel*D(i,j)
            do k=1,kmax
               y = min(1._rk, x*sqrt(max(0._rk, 0.5_rk*(SS(i,j,k)+SS(i,j,k-1)))))
               nu(i,j,k)=nu(i,j,k)+cSS*y
            end do
         end do
      end do
   end if
   end block shear

   ! 1.6.1 - small-cell limiter
   small_cell_limiter: block
   real(c_double) :: f, ihmin

   ihmin=1._rk/hmin
   do j=jmin,jmax
      do i=imin,imax
         if (mask(i,j) < 1) cycle

         do k=1,kmax
            f=2._rk * (haux(i,j,k)*ihmin - 1._rk)
            f = max(0._rk, min(1._rk, f))
            nu(i,j,k) = f*nu(i,j,k)
         end do
      end do
   end do
   end block small_cell_limiter

   ! 1.6.2 - shallow-water effect
   shallow_water_limiter: block
   real(c_double) :: relh

   if (chmin > ceps) then
      do j=jmin,jmax
         do i=imin,imax
            if (mask(i,j) < 1) cycle

            ! havg=Dmin(i,j)/kmax => hmin/havg = hmin*kmax/D(i,j)
            relh = 2._rk*(hmin*kmax/D(i,j) - 1._rk)
            if (relh < 0._rk .or. relh > 1._rk) then
               relh=(0.5_rk+sign(0.5_rk,relh))
            end if

            nu(i,j,:)=nu(i,j,:)+chmin*relh
         end do
      end do
   end if
   end block shallow_water_limiter
#ifdef _PRINT_
   write(100,*) 'nu ',nu(50,1,:)
#endif

   ! 2. Vertical filtering - done in Python

   ! 3. Lateral filtering - done in Python

end subroutine c_update_adaptive

end module

#ifdef _KKKKK_
! A copy of the adaptive coordinate coordinates as implemented by Bjarne
! Büchmann.
#include "adaptive_coordinates_6.F90"
#endif
