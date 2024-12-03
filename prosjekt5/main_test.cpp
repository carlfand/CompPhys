#include <armadillo>
#include <iostream>
#include <complex>

#include "pde.hpp"

int main() {

	arma::cx_mat M(3, 3, arma::fill::eye);
	M(2, 0) = 1.0;
	arma::mat M2(3, 3, arma::fill::eye);

	Wave_fnc_system prob(M, M2, 0.1, 0.1);
	std::cout << "Index test. N = " << prob.get_N() << ". (i, j) = (2, 2) -> k: " << prob.multi_index_to_single(2, 2) << 
	". For k = 3: (i, j) = (" << std::get<0>(prob.single_index_to_multi(3)) << ", " << std::get<1>(prob.single_index_to_multi(3)) << ")." << std::endl;

	std::cout << prob;
	double k = 1.0;
	int n_events_minus_one = 100 * 100 - 1;
	double n_e_min_one = n_events_minus_one * 1.0;
	for(int i = n_events_minus_one; i > 0; i--) {
		k += 1.0 / (i / n_e_min_one);
	}
	std::cout << k << std::endl;

	Wave_fnc_system three_by_three(M, M2, 1.0, 1.0);
	std::cout << three_by_three.B_mat();
	std::cout << three_by_three.A_mat();

	arma::cx_mat M_four_by_four(4, 4, arma::fill::eye);
	arma::mat M2_four_by_four(4, 4, arma::fill::eye);
	Wave_fnc_system four_by_four(M_four_by_four, M2_four_by_four, 1.0, 1.0);
	std::cout << four_by_four.B_mat();
	std::cout << four_by_four.A_mat();


	bool bool_test = 1;
	bool bool_test2 = 2;
	bool bool_test3 = false;
	double pi = 3.14;
	std::cout << "bool_test = 1, bool_test * pi: " << bool_test * pi << "\nbool_test3 = false, bool_test3 * pi: " << bool_test3 * pi << std::endl;
	std::cout << "bool_test2 = 2, bool_test2 * pi: " << bool_test2 * pi << std::endl;
	return 0;
}