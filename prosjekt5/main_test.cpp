#include <armadillo>
#include <iostream>
#include <complex>

#include "pde.hpp"

int main() {

	arma::cx_mat M(3, 3, arma::fill::eye);
	M(2, 0) = 1.0;
	M(0, 2) = 1.0;
	arma::mat M2(3, 3, arma::fill::eye);

	Wave_fnc_system prob(M, M2, 0.1, 0.1);
	std::cout << "Index test. N = " << prob.get_N() << ". (i, j) = (2, 2) -> k: " << prob.multi_index_to_single(2, 2) << 
	". For k = 3: (i, j) = (" << std::get<0>(prob.single_index_to_multi(3)) << ", " << std::get<1>(prob.single_index_to_multi(3)) << ")." << std::endl;

	std::cout << prob;

	Wave_fnc_system three_by_three(M, M2, 1.0, 1.0);
	std::cout << three_by_three.B_mat();
	// std::cout << three_by_three.A_mat();

	arma::cx_mat M_four_by_four(4, 4, arma::fill::eye);
	M_four_by_four(3, 0) = 1.0;
	M_four_by_four(2, 1) = 0.5;
	arma::mat M2_four_by_four(4, 4, arma::fill::eye);
	Wave_fnc_system four_by_four(M_four_by_four, M2_four_by_four, 1.0, 1.0);
	// std::cout << four_by_four.B_mat();
	// std::cout << four_by_four.A_mat();


	bool bool_test = 1;
	bool bool_test2 = 2;
	bool bool_test3 = false;
	double pi = 3.14;
	std::cout << "bool_test = 1, bool_test * pi: " << bool_test * pi << "\nbool_test3 = false, bool_test3 * pi: " << bool_test3 * pi << std::endl;
	std::cout << "bool_test2 = 2, bool_test2 * pi: " << bool_test2 * pi << std::endl;

	// Testing that B_mat_times_u() behaves as expected.
	// B_mat() has been visually confirmed to make B of correct shape. Therefore, B u with arma's matrix multiplication should yield the correct result.
	// Comparing therefore B_mat() * u to B_mat_times_u()
	arma::cx_vec fast_vec = three_by_three.B_mat_times_u();
	arma::cx_vec arma_mult_vec = three_by_three.B_mat() * three_by_three.get_u_vec();
	std::cout << "Our own multiplication algorithm gives Bu = b: \n" << fast_vec.t();
	std::cout << "Armadillo multiplication gives B*u = b\n" << arma_mult_vec.t();
	std::cout << "Their difference is:\n" << (fast_vec - arma_mult_vec).t();

	// Doing the same for the 4x4-case:
	arma::cx_vec fast_4_by_4 = four_by_four.B_mat_times_u();
	arma::cx_vec arma_mult_4_by_4 = (four_by_four.B_mat() * four_by_four.get_u_vec());
	std::cout << "The difference between our mult.alg. and armadillo mult in the 4x4-case:\n" << (fast_4_by_4 - arma_mult_4_by_4).t();
	std::cout << "4 by 4 on matrix form:\n" << four_by_four.prob_vec_to_matrix();

	// Testing that Jacobi iteration behaves as expected
	// 3x3 first
	arma::cx_vec b = three_by_three.B_mat_times_u();
	arma::cx_vec u_next = three_by_three.jacobi_iteration(b);
	arma::cx_vec u_next_mat_mult = three_by_three.jacobi_iteration_mat_mult(b);
	std::cout << "A_no_diag:\n" << three_by_three.A_mat_no_diag();
	std::cout << "u_next:\n" << u_next.t();
	std::cout << "u_next_mat_mult:\n" << u_next_mat_mult.t();
	std::cout << "Difference between fast and mat mult jabcobi iteration:\n" << (u_next - u_next_mat_mult).t();

	// 4x4:
	arma::cx_vec b_four = four_by_four.B_mat_times_u();
	arma::cx_vec u_next_four = four_by_four.jacobi_iteration(b_four);
	arma::cx_vec u_next_mat_mult_four = four_by_four.jacobi_iteration_mat_mult(b_four);
	std::cout << "Difference between fast and mat mult jabcobi iteration, 4x4:\n" << (u_next_four - u_next_mat_mult_four).t();
	// Behaviour as expected.

	// Testing that jacobi method converges.
	Wave_fnc_system four_by_four_small(M_four_by_four, M2_four_by_four, 0.001, 0.001);
	print_sp_matrix_structure(four_by_four_small.B_sp_mat());
	print_sp_matrix_structure(four_by_four_small.A_sp_mat());
	arma::cx_vec u_next_small = four_by_four_small.time_step_gs(four_by_four_small.B_mat_times_u());
	arma::cx_vec u_next_small_jac = four_by_four_small.jacobi_iteration(four_by_four_small.B_mat_times_u());
	std::cout << "2-norm of probability_vector: " << arma::norm(four_by_four_small.get_u_vec()) << std::endl;
	std::cout << "2-norm of first gs attempt to solve A u^n+1 = b: " << arma::norm(u_next_small) << std::endl;
	std::cout << "2-norm of first jacobi attempt to solve A u^n+1 = b: " << arma::norm(u_next_small_jac) << std::endl;
	std::cout << four_by_four_small.prob_vec_to_matrix();
	four_by_four_small.time_step_arma();
	std::cout << four_by_four_small.prob_vec_to_matrix();

	// std::cout << four_by_four_small.prob_vec_to_matrix();
	// four_by_four_small.time_step_jacobi(pow(10, -6), 1000);
	// std::cout << four_by_four_small.prob_vec_to_matrix();

	return 0;
}