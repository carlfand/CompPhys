#ifndef pde_hpp
#define pde_hpp

#include <armadillo>

class Prob_density {
	private:
		arma::cx_mat density;
		// Assuming real potential.
		arma::mat potential;


	public:

		// Constructor.
		Prob_density(const arma::cx_mat& initial, const arma::mat& potential);



};


#endif