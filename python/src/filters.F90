module c_filters

   use iso_c_binding, only: c_int, c_double

   implicit none

contains

   subroutine c_horizontal_filter(imin, imax, jmin, jmax, kmax, halox, haloy, mask, w, var) bind(c)
      integer(c_int), intent(in), value :: imin, imax, jmin, jmax, kmax
      integer(c_int), intent(in), value :: halox
      integer(c_int), intent(in), value :: haloy
#define _A_  imin-halox:imax+halox,jmin-haloy:jmax+haloy
      integer(c_int), intent(in) :: mask(_A_)
      real(c_double), intent(in), value :: w
      real(c_double), intent(inout) :: var(_A_,kmax)
#undef _A_

      integer :: i, j, k
      integer :: rc
      real(c_double), allocatable :: x(:,:,:)
      integer :: n1, n2, n3, n4, n5

      if (w < 0. .or. w > 0.25) return

      allocate(x, source=var, stat=rc)
      if (rc /= 0) stop 'c_horizontal_filter: Error allocating x'
      do j = jmin, jmax
         do i = imin, imax
            if (mask(i,j) == 0) cycle
            n1=min(1,mask(i-1,j))
            n2=min(1,mask(i+1,j))
            n3=min(1,mask(j-1,j))
            n4=min(1,mask(j+1,j))
            n5=n1+n2+n3+n4
            var(i,j,:) = w*var(i,j,:)+(1.-w) &
                        *( n1*x(i-1,j,:) &
                          +n2*x(i+1,j,:) &
                          +n3*x(i,j-1,:) &
                          +n4*x(i,j+1,:))/n5
         end do
      end do
   end subroutine

   subroutine c_vertical_filter(nfilter, imin, imax, jmin, jmax, kmax, halox, haloy, mask, w, var) bind(c)
      integer(c_int), intent(in), value :: nfilter
      integer(c_int), intent(in), value :: imin, imax, jmin, jmax, kmax
      integer(c_int), intent(in), value :: halox
      integer(c_int), intent(in), value :: haloy
#define _A_  imin-halox:imax+halox,jmin-haloy:jmax+haloy
      integer(c_int), intent(in) :: mask(_A_)
      real(c_double), intent(in), value :: w
      real(c_double), intent(inout) :: var(_A_, kmax)
#undef _A_

      integer :: i, j, k, n

      real(c_double) :: col(0:kmax)
      real(c_double) :: w1, w2

      if (nfilter < 1 .or. (w < 0. .or. w > 1.)) return

      w1 = w
      w2 = (1-w)/2
      do n = 1, nfilter
         do j = jmin, jmax
            do i = imin, imax
               if (mask(i,j) == 0) cycle
               col(kmax) = var(i,j,kmax)
               col(1) = var(i,j,1)
               col(0) = col(1)
               do k = 2, kmax-1
                  col(k) = w1*col(k)+w2*(var(i,j,k-1)+var(i,j,k+1))
               end do
               var(i,j,:) = col(1:kmax)
            end do
         end do
      end do
   end subroutine

end module
