subroutine fastembedr_knn_euclidean_range(data, points, n_data, n_points, &
    n_features, k, exclude_self, query_start, query_end, indices, distances) &
    bind(C, name = "fastembedr_knn_euclidean_range")
  use iso_c_binding
  implicit none

  integer(c_int), value :: n_data
  integer(c_int), value :: n_points
  integer(c_int), value :: n_features
  integer(c_int), value :: k
  integer(c_int), value :: exclude_self
  integer(c_int), value :: query_start
  integer(c_int), value :: query_end
  real(c_double), intent(in) :: data(n_data, n_features)
  real(c_double), intent(in) :: points(n_points, n_features)
  integer(c_int), intent(out) :: indices(n_points, k)
  real(c_double), intent(out) :: distances(n_points, k)

  integer(c_int) :: q
  integer(c_int) :: i
  integer(c_int) :: c
  integer(c_int) :: j
  integer(c_int) :: pos
  real(c_double) :: y
  real(c_double) :: diff
  real(c_double) :: dist
  real(c_double), allocatable :: work(:)
  real(c_double), allocatable :: best_dist(:)
  integer(c_int), allocatable :: best_idx(:)

  if (query_end < query_start) return

  allocate(work(n_data))
  allocate(best_dist(k))
  allocate(best_idx(k))

  do q = query_start, query_end
    work = 0.0_c_double

    do c = 1, n_features
      y = points(q, c)
      do i = 1, n_data
        diff = data(i, c) - y
        work(i) = work(i) + diff * diff
      end do
    end do

    if (exclude_self /= 0 .and. q >= 1 .and. q <= n_data) then
      work(q) = huge(1.0_c_double)
    end if

    best_dist = huge(1.0_c_double)
    best_idx = huge(1_c_int)

    do i = 1, n_data
      dist = work(i)
      if (dist < best_dist(k) .or. &
          (dist == best_dist(k) .and. i < best_idx(k))) then
        pos = k
        do while (pos > 1 .and. &
            (dist < best_dist(pos - 1) .or. &
            (dist == best_dist(pos - 1) .and. i < best_idx(pos - 1))))
          best_dist(pos) = best_dist(pos - 1)
          best_idx(pos) = best_idx(pos - 1)
          pos = pos - 1
        end do
        best_dist(pos) = dist
        best_idx(pos) = i
      end if
    end do

    do j = 1, k
      indices(q, j) = best_idx(j)
      distances(q, j) = sqrt(max(best_dist(j), 0.0_c_double))
    end do
  end do

  deallocate(best_idx)
  deallocate(best_dist)
  deallocate(work)
end subroutine fastembedr_knn_euclidean_range
