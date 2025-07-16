program forrester_test

    use mpi
    implicit none
    real(kind=8) :: X, Y, Yt
    integer :: S
    real(kind=8) :: A, B, C
    integer :: irank, nproc, ierr
    character(len=255) :: fname_in, fname_out

    ! Set up MPI
    ! Note MPI is a bit of a farce here as there is nothing to parallelise
    call MPI_init(ierr)
    call MPI_comm_size(MPI_COMM_WORLD, nproc, ierr)
    call MPI_comm_rank(MPI_COMM_WORLD, irank, ierr)

    ! Get filenames from cmd line args
    call get_command_argument(1, fname_in)
    call get_command_argument(2, fname_out)

    if(irank .eq. 0) call readin(fname_in, X, S)

    ! Send X and S to all procs
    call MPI_bcast(X, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
    call MPI_bcast(S, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

    call forrester(X,Y)
    A = 1.0 - (1 - S) * 0.5
    B = (1 - S) * 10.0
    C = (1 - S) * 5.0
    Y = -(A * Y + B * (X - 0.5) + C)

    ! Gather summed info on root
    call MPI_reduce(Y, Yt, 1, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    ! Average over mpi processors
    Y = Yt / nproc

    if(irank .eq. 0) call writeout(fname_out, Y)

    contains

    subroutine forrester(x,y)
        implicit none
        real :: x
        real :: y

        y = (6 * x - 2) ** 2 * sin(12 * x + 4)

    end subroutine forrester

    subroutine readin(filename, X, S)
        implicit none
        character(len=*), intent(in) :: filename
        integer ::  unit
        real :: X
        integer :: S

        namelist /inputs/ X, S

        open (newunit=unit, file=trim(filename), status='old', action='read')

        read (unit=unit, nml=inputs)

        close(unit)

    end subroutine readin

    subroutine writeout(filename,y)
        implicit none
        real :: y
        character(len=*), intent(in) :: filename
        integer :: unit

        open (newunit=unit, file=trim(filename), action='write')
        write(unit,*) y
        close(unit)

    end subroutine writeout

end program forrester_test
