#ifndef pde_hpp
#define pde_hpp

#include <armadillo>
#include <complex>
#include <utility>
#include <iostream>

class Wave_fnc_system {
	private:
		arma::cx_mat density;		// Complex wave-function in 2 dimensions (N times N points).
		arma::cx_vec density_vec;	// Complex wave-function flattened into 1 dimesion (of length N²)
		// arma::mat potential;		// Potential. Assumed to be real.
		double h;					// Stepsize in spatial dimensions
		std::complex<double> r;		// Factors which appears in matrices
		double delta_t;				// Stepsize in time-dimension
		int N;						// The size of the density and potential, N x N-matrices.
		arma::cx_vec b_diag;		// Contains the diagonal elements of the B-matrix (A u^(n + 1) = B u^n)
		arma::cx_vec a_diag_inv;	// Contains 1 / a_kk, where a_kk are the diagonal elements of the A-matrix.

		// Functions to be called by the constructor:

		// For filling b_diag and a_diag_inv as they are supposed to. 
		// b_diag with 1 - 4r - i delta_t v_ij / 2 
		// a_diag_inv with 1 / (1 + 4r + i delta_t v_ij / 2 )
		void fill_diagonals(const arma::mat& potential);
		void fill_density_vec(const arma::cx_mat& initial);


	public:

		// Constructor.
		Wave_fnc_system(const arma::cx_mat& initial, const arma::mat& potential, const double h, const double delta_t);

		// Converting index of type (i, j) into k. "Flattening the two dim array".
		int multi_index_to_single(const int& i, const int& j);

		// Inverse of the above.
		std::pair<int, int> single_index_to_multi(const int& k);

		int get_N() {return N;};

		// Overloading << operator for Wave_fnc_system class. For quick access during debugging.
		friend std::ostream& operator<<(std::ostream& os, const Wave_fnc_system& prob);

		// Evolve density by one dt using Crank-Nicholson and Gauss-Seidel for the following set of equations Au^(n + 1) = Bu^(n)
		// eps is the convergence criterion for Gauss-Seidel.
		void time_evolution_gs(double eps=pow(10.0, -6));

		// Evolve density by one dt using Crank-Nicholson and Jacobi. This converges more slowly than Gauss-Seidel, but it is 
		// readily parallelised.
		void time_evolution_jacobi(double eps=pow(10.0, -6));

		// Returns the flattened vector on the shape of a N times N matrix. For writing to file.
		arma::cx_mat density_to_matrix();

		// There is no need to work explicitly with the matrices B and A. But the problems forces us to construct them either way,
		// so here we do so.
		arma::cx_mat B_mat();

		// As above.
		arma::cx_mat A_mat();
};	


#endif