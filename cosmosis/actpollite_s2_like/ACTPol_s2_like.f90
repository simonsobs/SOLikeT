module ACTPol_CMBonly 
  use Options
  implicit none
  private
  integer, parameter :: campc = KIND(1.d0)

  real(campc), dimension(:), allocatable ::  bval,X_data,X_sig,diff_vec
  real(campc), dimension(:,:), allocatable :: win_func_tt,win_func_te,win_func_ee
  real(campc), dimension(:,:), allocatable :: covmat, fisher, cov_tot
 
  public actpol_like_init, actpol_calc_like, get_free_lun
  contains
  
  ! ===========================================================================
  subroutine actpol_like_init
 
  integer  :: i,j,lun,il,info,dum,k,bin_no
  character(LEN=1024) :: like_file, cov_file, bbltt_file, bblte_file, bblee_file
  logical  :: good
    
  print *, 'Initializing ACTPol likelihood, version '//ap_like
 
  !b0 =4 !first bin in TT
!  nbintt = 42 !500-4000
!  nbinte = 45 !250-4000
!  nbinee = 45 !250-4000
    

  like_file    = trim(act_data_dir)//'cl_cmb_aps2.dat'
  cov_file     = trim(act_data_dir)//'c_matrix_actpol.dat'
  bbltt_file   = trim(act_data_dir)//'BblMean.dat'
  bblte_file   = trim(act_data_dir)//'BblMean_Cross.dat'
  bblee_file   = trim(act_data_dir)//'BblMean_Pol.dat'
 
  allocate(bval(nbin),X_data(nbin),X_sig(nbin))
  allocate(covmat(nbin,nbin),cov_tot(nbin,nbin))
   
  inquire(file=like_file, exist=good)
  if(.not.good)then
     write(*,*) 'file not found', trim(like_file), trim(act_data_dir)
     stop
  endif
   
  call get_free_lun(lun)
  open(unit=lun,file=like_file,form='formatted',status='unknown',action='read')
  do i=1,nbin !read TT,TE,EE
     read(lun,*) bval(i),X_data(i),X_sig(i)
  enddo
  close(lun)

  inquire(file=cov_file, exist=good)
  if(.not.good)then
     write(*,*) 'file not found', trim(cov_file), trim(act_data_dir)
     stop
  endif
  call get_free_lun(lun)
  open(unit=lun,file=cov_file,form='unformatted',status='old')
      read(lun) covmat
  close(lun)
  do i=1,nbin
     do j=i+1,nbin
        covmat(j,i)=covmat(i,j) !covmat is now full matrix
     enddo
  enddo

  call get_free_lun(lun)
  open(unit=lun,file=bbltt_file,form='formatted',status='unknown',action='read')
  allocate(win_func_tt(1:bmax,1:lmax_win)) !Defined over ACTPol's full ell range 
  do i=1,bmax
     read(lun,*) (win_func_tt(i,il), il=2,lmax_win)
  enddo
  close(lun)

  call get_free_lun(lun)
  open(unit=lun,file=bblte_file,form='formatted',status='unknown',action='read')
  allocate(win_func_te(1:bmax,1:lmax_win)) !Defined over ACTPol's full ell range 
  do i=1,bmax
     read(lun,*) (win_func_te(i,il), il=2,lmax_win)
  enddo
  close(lun)

  call get_free_lun(lun)
  open(unit=lun,file=bblee_file,form='formatted',status='unknown',action='read')
  allocate(win_func_ee(1:bmax,1:lmax_win)) !Defined over ACTPol's full ell range 
  do i=1,bmax
     read(lun,*) (win_func_ee(i,il), il=2,lmax_win)
  enddo
  close(lun)

  end subroutine actpol_like_init
  
  ! ===========================================================================
  subroutine actpol_calc_like(aplike,cell_tt,cell_te,cell_ee,yp)

  real(campc), dimension(2:) :: cell_tt,cell_ee,cell_te
  real(campc) :: cltt(2:lmax_win), clte(2:lmax_win), clee(2:lmax_win)
  real(campc) :: cl_tt(bmax),cl_te(bmax),cl_ee(bmax)
  real(campc) :: aplike, yp
  real(campc) :: tmp(nbin,1)
  integer :: bin_no,lun,il,i,j,info
  real(campc), allocatable ::  Y(:), X_model(:)
  real(campc), allocatable :: ptemp(:)

  if (.not. allocated(Y)) then
     allocate(X_model(nbin),Y(nbin))
     X_model = 0
     Y = 0
  end if

  cltt(2:lmax_win)=0.d0 !Neglect above tt_lmax
  clte(2:lmax_win)=0.d0 
  clee(2:lmax_win)=0.d0 


  call get_free_lun(lun)
  open(unit=lun,file='testcls_rh.dat',status='unknown',action='write')


!  do i=2,tt_lmax
     
!     write(lun,*) i, cell_tt(i), cell_te(i), cell_ee(i)
!  enddo

!  write(88,*), cell_tt
!  write(89,*), cell_ee
!  write(90,*), cell_te
!  stop

  do i=2,tt_lmax
     cltt(i)=cell_tt(i)/real(i)/real(i+1.d0)*2.d0*PI
     clte(i)=cell_te(i)/real(i)/real(i+1.d0)*2.d0*PI
     clee(i)=cell_ee(i)/real(i)/real(i+1.d0)*2.d0*PI
  end do
 

  cl_tt(1:bmax)=MATMUL(win_func_tt(1:bmax,2:lmax_win),cltt(2:lmax_win))
  cl_te(1:bmax)=MATMUL(win_func_te(1:bmax,2:lmax_win),clte(2:lmax_win))
  cl_ee(1:bmax)=MATMUL(win_func_ee(1:bmax,2:lmax_win),clee(2:lmax_win))

  X_model(1:nbintt) = cl_tt(b0:b0-1+nbintt) !TT
  X_model(nbintt+1:nbintt+nbinte) = cl_te(1:nbinte)*yp !TE
  X_model(nbintt+nbinte+1:nbintt+nbinte+nbinee) = cl_ee(1:nbinee)*yp**2.d0 !EE
 
  !Start basic chisq 
  Y = X_data - X_model


!  write(88,*) X_data
!  write(89,*) X_model
  !Inflate covmat with ACTPol temperature calibration
  tmp(:,:)=0.d0
  tmp(1:nbin,1) = X_model(1:nbin)
  cov_tot(:,:) = covmat(:,:)+sigc**2.d0*matmul(tmp,transpose(tmp))

!  print*, 'Using use_act_tt:', use_act_tt
!  print*, 'Using use_act_ee:', use_act_ee
!  print*, 'Using use_act_te:', use_act_te

  !Select data
  !Only TT
  if((use_act_tt .eqv. .true.) .and. (use_act_te .eqv. .false.) .and. (use_act_ee .eqv. .false.)) then
       bin_no=nbintt
       allocate(fisher(bin_no,bin_no),diff_vec(bin_no),ptemp(bin_no))
       diff_vec(:)=Y(1:bin_no)
       fisher(:,:)=cov_tot(1:bin_no,1:bin_no)
  !Only TE
  else if((use_act_tt .eqv. .false.) .and. (use_act_te .eqv. .true.) .and. (use_act_ee .eqv. .false.)) then 
       bin_no=nbinte
       allocate(fisher(bin_no,bin_no),diff_vec(bin_no),ptemp(bin_no))
       diff_vec(:)=Y(nbintt+1:nbintt+bin_no)
       fisher(:,:)=cov_tot(nbintt+1:nbintt+bin_no,nbintt+1:nbintt+bin_no)
  !Only EE
  else if((use_act_tt .eqv. .false.) .and. (use_act_te .eqv. .false.) .and. (use_act_ee .eqv. .true.)) then
       bin_no=nbinee
       allocate(fisher(bin_no,bin_no),diff_vec(bin_no),ptemp(bin_no))
       diff_vec(:)=Y(nbintt+nbinte+1:nbintt+nbinte+bin_no)
       fisher(:,:)=cov_tot(nbintt+nbinte+1:nbintt+nbinte+bin_no,nbintt+nbinte+1:nbintt+nbinte+bin_no)
  !All
  else if ((use_act_tt .eqv. .true.) .and. (use_act_te .eqv. .true.) .and. (use_act_ee .eqv. .true.)) then
       bin_no=nbin
       allocate(fisher(nbin,nbin),diff_vec(nbin),ptemp(nbin))
       diff_vec(:)=Y(:)
       fisher(:,:)=cov_tot(:,:)
  else
     write(*,*) 'Fail: no possible options chosen, please rechoose your TT,EE,TE selection'
  endif

  !Invert covmat
  call dpotrf('U',bin_no,fisher,bin_no,info)

  if(info.ne.0)then
 !    print*, 'hi erminia'
     print*, ' info in dpotrf =', info
     stop
  endif

  call dpotri('U',bin_no,fisher,bin_no,info)
  if(info.ne.0)then
     print*, ' info in dpotri =', info
     stop
  endif
  do i=1,bin_no
     do j=i,bin_no
        fisher(j,i)=fisher(i,j)
     enddo
  enddo


  ptemp=matmul(fisher,diff_vec)
  aplike=sum(ptemp*diff_vec)
  aplike = aplike/2.d0
 
  deallocate(X_model,Y)
  deallocate(fisher,diff_vec,ptemp)
    
 end subroutine actpol_calc_like
  
 subroutine get_free_lun(lun)

 implicit none
 integer, intent(out) :: lun
 integer, save :: last_lun = 19  
 logical :: used
 lun = last_lun
 do
   inquire( unit=lun, opened=used )
   if ( .not. used ) exit
       lun = lun + 1
 end do
    
 last_lun = lun
 end subroutine get_free_lun

end module ACTPol_CMBonly
