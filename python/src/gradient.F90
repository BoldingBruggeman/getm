module c_gradient

   use iso_c_binding, only: c_int, c_double

   implicit none

contains

   subroutine c_gradient_x(nx1, ny1, nx2, ny2, nz, source, idx, target, ioffset, joffset) bind(c)
      integer(c_int), intent(in), value :: nx1, ny1, nx2, ny2, nz, ioffset, joffset
      real(c_double), intent(in)        :: source(nx1, ny1, nz), idx(nx2, ny2)
      real(c_double), intent(inout)     :: target(nx2, ny2, nz)

      integer :: i, j, k

      do k = 1, nz
         do j = 1, ny1
            do i = 1, nx1 - 1
               target(i + ioffset, j + joffset, k) = (source(i + 1, j, k) - source(i, j, k)) * idx(i + ioffset, j + joffset)
            end do
         end do
      end do
   end subroutine

   subroutine c_gradient_y(nx1, ny1, nx2, ny2, nz, source, idy, target, ioffset, joffset) bind(c)
      integer(c_int), intent(in), value :: nx1, ny1, nx2, ny2, nz, ioffset, joffset
      real(c_double), intent(in)        :: source(nx1, ny1, nz), idy(nx2, ny2)
      real(c_double), intent(inout)     :: target(nx2, ny2, nz)

      integer :: i, j, k

      do k = 1, nz
         do j = 1, ny1 - 1
            do i = 1, nx1
               target(i + ioffset, j + joffset, k) = (source(i, j + 1, k) - source(i, j, k)) * idy(i + ioffset, j + joffset)
            end do
         end do
      end do
   end subroutine

end module
