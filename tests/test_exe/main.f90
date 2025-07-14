program forrester_test

    use mpi
    implicit none
    real, dimension(:), allocatable :: Xs, Ys, Ss
    integer :: n

    call mpi_init()

    A = 1.0 - (1 - Ss) * 0.5
    B = (1 - Ss) * 10.0
    C = (1 - Ss) * 5.0

    call forrester(Xs,ys)
    ys = -(A * ys + B * (Xs - 0.5) + C)

    contains

    subroutine forrester(x,y)
        real, dimension(:), intent(in) :: x
        real, dimension(:), intent(out) :: y

        y = (6 * x - 2) ** 2 * sin(12 * x + 4)

    end subroutine forrester

    subroutine readin()

    end subroutine readin

    subroutine writeout()

    end subroutine writeout

end program forrester_test
