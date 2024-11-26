#include <armadillo>

#include "pde.hpp"

Prob_density::Prob_density(const arma::cx_mat& initial, const arma::mat& potential) {

	this->density = initial;
	this->potential = potential;
	
}