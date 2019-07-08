! ===========================================================================
MODULE OPTIONS
! This module contains the options and settings used in the likelihood code
! ===========================================================================

!Options
!--------------------------------------------------------------
! location of  data
! -------------------------------------------------------------
#ifdef ACTPOL_DATA_DIR
  character(len=5000) :: act_data_dir =&
ACTPOL_DATA_DIR
#else 
  character(len=5000) :: act_data_dir = ''
#endif 

!  character(len=*), parameter :: data_dir = $
!  character(len=*),parameter:: ACT_data_dir = 'data_act/'
  character(LEN=*), parameter, public :: ap_like='ACTPol_s2_cmbonly_like' 

!--------------------------------------------------------------
! change these to include/exclude observables 
!--------------------------------------------------------------
  logical :: use_act_tt  = .true.  ! use TT only
  logical :: use_act_te = .true.  ! use TE+EE only
  logical :: use_act_ee  = .true. ! use EE only, false dy default and not to use in combination

!--------------------------------------------------------------
!Settings (should not be altered)
!--------------------------------------------------------------
! general settings
!--------------------------------------------------------------
  integer :: tt_lmax = 6000    
  real(8), parameter :: pi = 3.14159265358979323846264d0

!----------------------------------------------------------------
! likelihood terms from ACT data
!----------------------------------------------------------------
  integer, parameter :: nbintt = 42  !max nbins in ACT TT data  
  integer, parameter :: nbinte = 45  !max nbins in ACT TE data
  integer, parameter :: nbinee = 45  !max nbins in ACT EE data
  integer, parameter :: nbin   = 132 !total bins
  integer, parameter :: lmax_win = 9000 !total ell in window functions     
  integer, parameter :: bmax  = 53   !number of bins in full window function
  integer, parameter :: b0     = 4    !First bin of full window function used  
  real(8), parameter :: sigc = 0.01d0 !overall calibration uncertainty     

END MODULE OPTIONS
! ===========================================================================
