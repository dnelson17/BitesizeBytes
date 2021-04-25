module precisn
    implicit none
    INTEGER, PARAMETER    :: wp = SELECTED_REAL_KIND(6)  ! `single' precision

end module precisn

program parallel_matrix
    USE precisn
    USE MPI
    implicit none
    real(wp), allocatable :: my_M(:,:)
    real(wp), allocatable :: my_V(:,:)
    real(wp), allocatable :: my_R(:,:)
    integer               :: ierr, ii, jj
    integer               :: ncpus
    integer               :: myrank
    integer               :: m_size
    integer               :: numrows
    real(wp)              :: start_time, end_time
    real(wp), parameter   :: alpha = 1.0_wp
    real(wp), parameter   :: beta  = 0.0_wp
    real(wp)              :: sumtime
    integer, parameter    :: niter=10
    
    call MPI_Init(ierr)
    call MPI_comm_size(mpi_comm_world, ncpus, ierr)
    call MPI_comm_rank(mpi_comm_world, myrank, ierr)

    do jj=10,15
        m_size = 2**jj

        numrows = m_size/ncpus

        call setup_matrix(numrows, m_size)
        
        call MPI_Barrier(mpi_comm_world, ierr)
        start_time = hel_time()

        do ii=1,niter
            call sgemm( 'n','n', m_size, m_size, numrows,  alpha, my_M, m_size, my_V, numrows, beta, my_R,m_size )
        end do

        call MPI_Barrier(mpi_comm_world, ierr)
        end_time   = hel_time()

        if (myrank==0) then
            print *, m_size, (end_time - start_time)/niter
        end if

        call tear_down_matrix
    end do

    call MPI_Finalize(ierr)



contains

subroutine setup_matrix (nrows,m_size)
    implicit none
    integer, intent(in) :: nrows
    integer, intent(in) :: m_size
    integer             :: ierr, ii, jj

    allocate(my_M(m_size,nrows),stat=ierr)
    if (ierr /= 0) print *, "allocation error"
    allocate(my_V(nrows,m_size))
    allocate(my_R(m_size,m_size))

    call random_number(my_M)
    call random_number(my_V)
    my_R = 0.0_wp

end subroutine setup_matrix

subroutine tear_down_matrix
    implicit none 
    integer :: ierr

    deallocate(my_M, my_V, my_R, stat=ierr)
    if (ierr /= 0) print *, "allocation error"

end subroutine tear_down_matrix

REAL(wp) FUNCTION hel_time()

    INTEGER :: it0, count_rate

    it0 = 0
    count_rate = 1

    CALL SYSTEM_CLOCK(it0, count_rate)
    hel_time = REAL(it0, wp) / REAL(count_rate, wp)

END FUNCTION hel_time
end program parallel_matrix
