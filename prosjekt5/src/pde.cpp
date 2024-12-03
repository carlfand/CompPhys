#include <armadillo>
#include <utility>
#include <vector>
#include <assert.h>
#include <iostream>

#include "pde.hpp"

Wave_fnc_system::Wave_fnc_system(const arma::cx_mat& initial, const arma::mat& potential, const double h, const double delta_t) {
	// Constructor if Wave_fnc_system class. First, assigning a few values.
	this->density = initial;
	this->h = h;
	this->delta_t = delta_t;
	std::complex<double> imag(0.0, 1.0);
	r = imag * delta_t / (2.0 * pow(h, 2));
	N = density.n_rows;			// Assuming here square matrix, i.e. density.n_rows = density.n_cols.
	b_diag = arma::cx_vec(N * N);		// Constructing arma vector of correct size
	a_diag_inv = arma::cx_vec(N * N);	// As above.
	fill_diagonals(potential);			// Filling the arma vector accordingly	

	density_vec = arma::cx_vec(N * N);	// Initialising density vec of right size.
	fill_density_vec(initial);			// Filling density_vec as it should.
}

int Wave_fnc_system::multi_index_to_single(const int& i, const int& j) {
	// i and j runs from 0 to N-1. 
	return i + j * (this->N);
}

std::pair<int, int> Wave_fnc_system::single_index_to_multi(const int& k) {
	return {k % N, k / (N - 1)};
}

void Wave_fnc_system::fill_diagonals(const arma::mat& potential) {
	// Diagonal of B contains 1 - 4 r - i delta_t v_ij / 2
	// Diagonal of A contains 1 + 4 r + i delta_t v_ij / 2
	int k = 0;
	std::complex<double> b_first_terms = 1.0 - 4.0 * r;
	std::complex<double> a_first_terms = 1.0 + 4.0 * r;
	std::complex<double> last_term;
	std::complex<double> imag(0.0, 1.0);
	std::complex factor = imag * delta_t / 2.0;
	// std::cout << "Re(1 - 4 r): " << b_first_terms.real() << "\nIm(1 - 4 r): " << b_first_terms.imag() << std::endl;
	// Column major double loop:
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			// For testing purposes:
			// assert(("There is an indexing error. Fix the code logic!", k == multi_index_to_single(i, j)));
			last_term = factor * potential(i, j);	// Corresponds to i delta_t v_ij / 2
			// Setting diagonal element k:
			b_diag(k) = b_first_terms - last_term;
			// Strangely, it seems that printing cx_vec to terminal gives the opposite sign to printing them elementwise here.
			// std::cout << b_diag(k).imag() << "    ";
			a_diag_inv(k) = 1.0 / (a_first_terms + last_term);
			k += 1;		// Incrementing k by 1 here gives the correct single index.
		}
	}
}

void Wave_fnc_system::fill_density_vec(const arma::cx_mat& initial) {
	int k = 0;
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			density_vec(k) = initial(i, j);
			k += 1;
		}
	}
}

std::ostream& operator<<(std::ostream& os, const Wave_fnc_system& prob) {
	os << "B diagonal: \n" << prob.b_diag.t() << "A diagonal inverse: \n" << prob.a_diag_inv.t() << std::endl;
	os << "u_ij = \n" << prob.density << std::endl;
	os << "u_vec = \n" << prob.density_vec.t() << std::endl;
	return os;
}

void Wave_fnc_system::time_evolution_gs(double eps) {
	// Calculating first B u = c.
	arma::cx_vec intermediate(N * N);
	// Consider c(k). k, n in {0, ..., N² - 1}. There are potentially four values the matrix-vector B_{k, n}u_n 
	// multiplication might "hit":
	// r at B_{k, k - N}, r at B_{k, k - 1}, b_diag(k) at B_{k, k} (this one is always hit), r at B_{k, k + 1}
	// or r at B_{k, k + N}
	// k = 0: only the three last are "hit".
	intermediate(0) = b_diag(0) * density_vec(0) + r * density_vec(1) + r * density_vec(N);
	// TODO: Huske på de stedene det IKKE skal være r. Finn en lur måte å la løkken løpe igjennom det på.
	// Idé: Kanskje det enkleste er å behandle det som om alle r er der, og så løpe gjennom én gang til og trekke fra de stedene det skulle være relevant.
	// Dette sparer If-statements, som er bra for kompilator-optimalisering.
}

void Wave_fnc_system::time_evolution_jacobi(double eps) {
	// Denne har fordelen at den kan parallelliseres. Dersom den går 6x så fort, kan den konvergere 6 x saktere med samme kjøretid.
}

arma::cx_mat Wave_fnc_system::density_to_matrix() {
	// TODO!
}

arma::cx_mat Wave_fnc_system::B_mat() {
	int N_square = N * N;
	arma::cx_mat B(N_square, N_square);
	// Filling diagonal.
	for(int k = 0; k < N_square; k++) {
		B(k, k) = b_diag(k);
	}
	// Filling sub and superdiagonal
	for(int k = 0; k < N_square - 1; k++){
		B(k, k + 1) = r;
		B(k + 1, k) = r;
	}
	// Removing r from the places where it should.
	for(int k = 0; k < N - 1; k++) {
		B(N + N * k - 1, N + N * k) -= r;
		B(N + N * k, N + N * k - 1) -= r;
	}
	// Filling the sub-N-diagonal and super-N-diagonal:
	for(int k = 0; k < N_square - N; k++) {
		B(k, k + N) = r;
		B(k + N, k) = r;
	}
	return B;

}

arma::cx_mat Wave_fnc_system::A_mat() {
	int N_square = N * N;
	arma::cx_mat A(N_square, N_square);
	// Filling diagonal.
	for(int k = 0; k < N_square; k++) {
		A(k, k) = 1.0 / a_diag_inv(k);
	}
	// Filling sub and superdiagonal
	for(int k = 0; k < N_square - 1; k++){
		A(k, k + 1) = -r;
		A(k + 1, k) = -r;
	}
	// Removing r from the places where it should.
	for(int k = 0; k < N - 1; k++) {
		A(N + N * k - 1, N + N * k) += r;
		A(N + N * k, N + N * k - 1) += r;
	}
	// Filling the sub-N-diagonal and super-N-diagonal:
	for(int k = 0; k < N_square - N; k++) {
		A(k, k + N) = -r;
		A(k + N, k) = -r;
	}
	return A;
}
