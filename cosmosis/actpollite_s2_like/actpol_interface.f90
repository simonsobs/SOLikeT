module ACTPolMod
implicit none
contains 

function actpol_initial_setup(block) result(status)
    use cosmosis_modules
    use options
    use ACTParams
    integer(c_size_t) :: block
    integer status
!    type(my_settings), pointer :: settings
    character(*), parameter :: actpol_section = option_section


    status=0
  
    status = status + datablock_get_int_default(block, actpol_section, "act_tt_lmax", 6000, tt_lmax)
    status = status + datablock_get_logical_default(block, actpol_section, "use_act_tt", .true., use_act_tt) 
    status = status + datablock_get_logical_default(block, actpol_section, "use_act_ee", .true., use_act_ee) 
    status = status + datablock_get_logical_default(block, actpol_section, "use_act_te", .true., use_act_te) 


    if (status .ne. 0) then
        write(*,*) "Error setting up act pol likelihood."
        write(*,*) "You need to set these parameters in the ini file:"
        write(*,*) "ac_ttt_lmax", tt_lmax
        write(*,*) "use_act_ee", use_act_ee
        write(*,*) "use_act_te", use_act_te
        write(*,*) "use_act_tt", use_act_tt
        stop
    endif

end function actpol_initial_setup

function actpol_set_params(block, afg) result(status)
    use cosmosis_modules
    use ACTParams
    integer (c_int) :: status
    integer (c_size_t) :: block  ! <--- no "value" here - being called from fortran no C
    type(actpol_fg) :: afg
    character(*), parameter :: actpol_section = "actpol" ! this is in the values file, and must always be called actpol explicitly
    real(8):: tmp 

    tmp = afg%yp

    status=0
    status = status + datablock_get_double(block, actpol_section, "yp", afg%yp)

    if (status .ne. 0) then
        write(*,*) "Error setting up act pol parameters."
        write(*,*) "You need to set these parameters in the values.ini file:"
        stop
     endif

end function actpol_set_params

end module ACTPolMod


function setup(block) result(status)
    use cosmosis_modules
    Use ACTParams
    use ACTPolMod
    use options
    implicit none
    integer(c_size_t), value :: block
    integer status
    
    status =  actpol_initial_setup(block)
    status=0
end function

function execute(block, config) result(status)
    use Options
    use cosmosis_modules
    use ACTPol_CMBonly
    use ACTParams
    use ACTPolMod

    implicit none

!    real(8), dimension(:,:), allocatable :: cell
    real(8), allocatable, dimension(:) :: cell_tt,cell_ee,cell_te
    integer(4), allocatable, dimension(:) :: ell
    real(8)  ::  aplike
    integer n_ell
    integer(cosmosis_status) :: status
    integer(cosmosis_block), value :: block
    integer(c_size_t) :: config
    type(actpol_fg) :: afg
    integer :: bin_no,lun,il,i,j,info
    integer ell_start, cc
    real(8), allocatable ::  Y(:), X_model(:)
    logical :: actpol_init=.true.
    logical :: rhtest

    rhtest = .true.
    status=0

    ! Loading in the initial parameters and the data
    status= actpol_set_params(block, afg) 


    !Load all the columns 
    n_ell=0
    status = status + datablock_get_int_array_1d(block, cmb_cl_section, "ELL", ell, n_ell)
    status = status + datablock_get_double_array_1d(block, cmb_cl_section, "TT",  cell_tt,  n_ell)
    status = status + datablock_get_double_array_1d(block, cmb_cl_section, "EE",  cell_ee,  n_ell)
    status = status + datablock_get_double_array_1d(block, cmb_cl_section, "TE",  cell_te,  n_ell)


    !Check for errors in loading the columns.  Free any allocated mem if so
    if (status .ne. 0 .or. n_ell .le. 0) then
!        if (allocated(cell)) deallocate(cell)
        if (allocated(ell)) deallocate(ell)
        if (allocated(cell_tt)) deallocate(cell_tt)
        if (allocated(cell_ee)) deallocate(cell_ee)
        if (allocated(cell_te)) deallocate(cell_te)
        status = max(status, 1)
        return
    endif
        
    !find ell=2 - the column may have started at ell=0 or ell=1
    do ell_start=1,n_ell
        if (ell(ell_start)==2) exit
    enddo
    
    
    !If the ells are wrong and cannot find 2 then complain and free the memory
    if (ell_start .ge. n_ell) then
        write(*,*) "Could not figure out where ell=2 was in the theory"
        status=2
        !if (allocated(cell)) deallocate(cell)
        if (allocated(ell)) deallocate(ell)
        if (allocated(cell_tt)) deallocate(cell_tt)
        if (allocated(cell_ee)) deallocate(cell_ee)
        if (allocated(cell_te)) deallocate(cell_te)
        return
    endif

    
    if (actpol_init .eqv. .true.)  then
       call actpol_like_init ! this initialises both the data and the foreground vectors                                                
       actpol_init = .false.
    endif


    aplike=0.0d0

    call  actpol_calc_like(aplike,cell_tt,cell_te,cell_ee,afg%yp)
    
    !Free memory
    if (allocated(ell)) deallocate(ell)
    if (allocated(cell_tt)) deallocate(cell_tt)
    if (allocated(cell_ee)) deallocate(cell_ee)
    if (allocated(cell_te)) deallocate(cell_te)
    
    
    status = datablock_put_double(block, likelihoods_section, "ACTPOL_LIKE", aplike)
    
    cc=0                                                                                                                                                                                       
        
if (rhtest) then

   write(*,*) '-------------------------------------'
   write(*,*) 'ACTPol chi2 = ', 2*aplike
   write(*,*) 'Expected = 147.747797921459'
   write(*,*) '-------------------------------------'

end if

end function execute
