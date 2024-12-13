module m_filters

   use iso_c_binding, only: c_int, c_double

   implicit none

contains

   subroutine c_horizontal_filter(nx, ny, nz, halox, haloy, &
                                  mask, w, var) bind(c)

     !! Simple horizontal filter where the central value is weighted by
     !! w and the avarage of the - active - neighbors by 1-w.

      integer(c_int), intent(in), value :: nx, ny, nz
      integer(c_int), intent(in), value :: halox, haloy
#define _D2_  -halox+1:nx+halox,-haloy+1:ny+haloy
      integer(c_int), intent(in) :: mask(_D2_)
      real(c_double), intent(in), value :: w
      real(c_double), intent(inout) :: var(_D2_,0:nz)
#undef _D2_

      real(c_double), allocatable :: x(:,:,:)
      integer :: n1, n2, n3, n4, n5
      integer :: rc
      integer :: imin=1, jmin=1, imax, jmax, kmax
      integer :: i, j, k

      if (w < 0._c_double .or. w > 0.25_c_double) return

      imax=nx; jmax=ny; kmax=nz

      allocate(x, source=var, stat=rc)
      if (rc /= 0) stop 'c_horizontal_filter: Error allocating x'
      do j = jmin, jmax
         do i = imin, imax
            if (mask(i,j) < 1) cycle

            n1=min(1,mask(i-1,j))
            n2=min(1,mask(i+1,j))
            n3=min(1,mask(j-1,j))
            n4=min(1,mask(j+1,j))
            n5=n1+n2+n3+n4
            var(i,j,1:kmax-1) = w*var(i,j,1:kmax-1)+(1._c_double-w) &
                        *( n1*x(i-1,j,1:kmax-1) &
                          +n2*x(i+1,j,1:kmax-1) &
                          +n3*x(i,j-1,1:kmax-1) &
                          +n4*x(i,j+1,1:kmax-1))/n5
         end do
      end do
   end subroutine

   subroutine c_vertical_filter(nx, ny, nz, halox, haloy, &
                                nfilter, mask, w, var) bind(c)

     !! Simple vartical filter where the central value is weighted by
     !! w and the two neighbors by 1-w/2.
     !! The filter will be applied nfilter times.

      integer(c_int), intent(in), value :: nx, ny, nz
      integer(c_int), intent(in), value :: halox, haloy
      integer(c_int), intent(in), value :: nfilter
#define _D2_  -halox+1:nx+halox,-haloy+1:ny+haloy
      integer(c_int), intent(in) :: mask(_D2_)
      real(c_double), intent(in), value :: w
      real(c_double), intent(inout) :: var(_D2_, 0:nz)
#undef _D2_

      real(c_double) :: col(0:nz)
      real(c_double) :: w1
      integer :: imin=1, jmin=1, imax, jmax, kmax
      integer :: i, j, k, n

!----------------------------------------------------------------------
      imax=nx; jmax=ny; kmax=nz

      if (nfilter < 1 .or. (w < 0._c_double .or. w > 1._c_double)) return

      w1 = (1_c_double-w)/2
      do n = 1, nfilter
         do j = jmin, jmax
            do i = imin, imax
               if (mask(i,j) < 1) cycle

               !col(kmax) = var(i,j,kmax)
               !col(kmin) = var(i,j,kmin)
               col(1) = var(i,j,1)
               do k = 2, kmax-2
                  col(k) = w*col(k)+w1*(var(i,j,k-1)+var(i,j,k+1))
               end do
               col(kmax-1) = var(i,j,kmax-1)
               var(i,j,1:kmax-1) = col(1:kmax-1)
            end do
         end do
      end do
   end subroutine

end module
