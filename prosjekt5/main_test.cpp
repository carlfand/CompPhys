#include <armadillo>
#include <iostream>
#include <complex>

#include "pde.hpp"

int main() {

	arma::cx_mat M(3, 3, arma::fill::eye);
	M(2, 0) = 1.0;
	arma::mat M2(3, 3, arma::fill::eye);

	Prob_density prob(M, M2, 0.1, 0.1);
	std::cout << "Index test. N = " << prob.get_N() << ". (i, j) = (2, 2) -> k: " << prob.multi_index_to_single(2, 2) << 
	". For k = 3: (i, j) = (" << std::get<0>(prob.single_index_to_multi(3)) << ", " << std::get<1>(prob.single_index_to_multi(3)) << ")." << std::endl;

	std::cout << prob;
	double k = 1.0;
	int n_events_minus_one = 100 * 100 - 1;
	double n_e_min_one = n_events_minus_one * 1.0;
	for(int i = n_events_minus_one; i > 0; i--) {
		k += 1.0 / (i / n_e_min_one);
	}
	std::cout << k;
	return 0;
}