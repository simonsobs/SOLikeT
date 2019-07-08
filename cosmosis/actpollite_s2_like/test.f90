!===========================================================
program test

!===========================================================
!E. Calabrese, J. Dunkley 2016
!===========================================================

use ACTPol_CMBonly 
use Options
implicit none
real(8), dimension(:), allocatable :: cell_tt,cell_ee,cell_te
character(LEN=128) :: filename
real(8)            :: aplike, yp
integer            :: lun, il, dummy

!---------------------------------------------------

print *,""
print *,"ACTPol CMB-only likelihood test"
print *,"==================================="
call actpol_like_init
!---------------------------------------------------
! read in test Cls
!---------------------------------------------------
filename = trim(act_data_dir)//'planck2015.dat'
write(*,*)"Reading in Cls from: ",trim(filename)

call get_free_lun(lun)
allocate(cell_tt(2:tt_lmax),cell_ee(2:tt_lmax),cell_te(2:tt_lmax)) 
cell_tt(2:tt_lmax)=0.d0
cell_ee(2:tt_lmax)=0.d0
cell_te(2:tt_lmax)=0.d0

open(unit=lun,file=filename,action='read',status='old')
do il=2,tt_lmax
   read(lun,*)dummy,cell_tt(il),cell_te(il),cell_ee(il)
enddo
close(lun)

yp =1.d0

call actpol_calc_like(aplike,cell_tt,cell_te,cell_ee,yp)

write(*,*) '-------------------------------------'
write(*,*) 'ACTPol chi2 = ', 2*aplike
write(*,*) 'Expected = 147.747797921459'
write(*,*) '-------------------------------------'

end program test
