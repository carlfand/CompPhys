#include <armadillo>
#include <utility>
#include <vector>
#include <assert.h>
#include <iostream>

#include "pde.hpp"

//////////////////////////////////////////////////////////////////
// Constructors and basic class methods

Wave_fnc_system::Wave_fnc_system(const arma::cx_mat& initial, const arma::mat& potential, const double h, const double delta_t) {
	// Constructor if Wave_fnc_system class. First, assigning a few values.
	this->h = h;
	this->delta_t = delta_t;
	std::complex<double> imag(0.0, 1.0);
	r = imag * delta_t / (2.0 * pow(h, 2));
	N = initial.n_rows;			// Assuming here square matrix, i.e. initial.n_rows = initial.n_cols.
	b_diag = arma::cx_vec(N * N);		// Constructing arma vector of correct size
	a_diag_inv = arma::cx_vec(N * N);	// As above.
	fill_diagonals(potential);			// Filling the arma vector accordingly	

	probability_vec = arma::cx_vec(N * N);	// Initialising probability_vec of right size.
	fill_probability_vec(initial);			// Filling probability_vec as it should.

	A_sp = A_sp_mat();
	B_sp = B_sp_mat();
}


Wave_fnc_system::Wave_fnc_system(const Wave_fnc_system& initial) {
	this->probability_vec = initial.probability_vec;
	this->h = initial.h;
	this->r = initial.r;		
	this->delta_t = initial.delta_t;
	this->N = initial.N;		
	this->b_diag = initial.b_diag;
	this->a_diag_inv = initial.a_diag_inv;
	this->A_sp = initial.A_sp;		
	this->B_sp = initial.B_sp;
}


int Wave_fnc_system::multi_index_to_single(const int& i, const int& j) {
	// i and j runs from 0 to N-1. 
	return i + j * (this->N);
}


std::pair<int, int> Wave_fnc_system::single_index_to_multi(const int& k) {
	return {k % N, k };
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


void Wave_fnc_system::fill_probability_vec(const arma::cx_mat& initial) {
	int k = 0;
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			probability_vec(k) = initial(i, j);
			k += 1;
		}
	}
}


std::ostream& operator<<(std::ostream& os, const Wave_fnc_system& prob) {
	os << "B diagonal: \n" << prob.b_diag.t() << "A diagonal inverse: \n" << prob.a_diag_inv.t() << std::endl;
	os << "u_ij = \n" << prob.prob_vec_to_matrix() << std::endl;
	// os << "u_vec = \n" << prob.probability_vec.t() << std::endl;
	return os;
}


//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Functionality for solving A u ^ (n + 1) = B u ^ n without in-built arma-solvers.


arma::cx_vec Wave_fnc_system::B_mat_times_u() {
	// Calculating B u^n = b. (supress n from here)
	// In component form: B_{lk} u_{k} = b_{l}.
	// Consider b(l). l, k in {0, ..., N² - 1}. There are potentially five values the matrix-vector B_{l, k}u_{k} multiplication might "hit":
	// 1) r at B_{k, k - N}, 
	// 2) r at B_{k, k - 1}, 
	// 3) b_diag(k) at B_{k, k} (this one is always hit), 
	// 4) r at B_{k, k + 1}
	// 5) r at B_{k, k + N}
	// Splitting the assignments for the different cases to avoid if-statements, in an attempt to aid compiler optimisation.

	int N_square = N * N;
	arma::cx_vec b(N_square);

	// k = 0: only the three last are "hit" [3) - 5)].
	b(0) = b_diag(0) * probability_vec(0) + r * (probability_vec(1) + probability_vec(N));
	// k = N * N - 1 (the last index), when only the three first are "hit" [1) - 3)].
	b(N_square - 1) = b_diag(N_square - 1) * probability_vec(N_square - 1) + r * (probability_vec(N_square - 2) + probability_vec(N_square - 1 - N));

	// The bulk, i.e. k in [N, N_square - N - 1], inclusive. [1) - 5)]
	// For some elements along the sub- and super-diagonal, there is a 0 instead of r. Handling those in the next, shorter loop.
	for(int k = N; k < N_square - N; k++){
		b(k) = b_diag(k) * probability_vec(k) + r * (probability_vec(k - N) + probability_vec(k - 1) + probability_vec(k + 1) + probability_vec(k + N));
	}

	// Handling k > N_square - 1 - N [which includes 1) - 4)] and k < N [which includes 2) - 5)].
	// Also handling the entries along the sub and super-diagonal where there is 0 instead of r.
	for(int k = 1; k < N; k++) {
		b(k) = b_diag(k) * probability_vec(k) + r * (probability_vec(k - 1) + probability_vec(k + 1) + probability_vec(k + N));
		b(N_square - 1 - k) = b_diag(N_square - 1 - k) * probability_vec(N_square - 1 - k) + r * (probability_vec(N_square - 2 - k) + probability_vec(N_square - k) + probability_vec(N_square - 1 - k - N));
		// sub- and super-diagonal
		b(k * N - 1) -= r * probability_vec(k * N);
		b(k * N) -= r * probability_vec(k * N - 1);
	}
	return b;
}

//////////////////////////////////
// Jacobi iteration method


arma::cx_vec Wave_fnc_system::jacobi_iteration(const arma::cx_vec& b, const arma::cx_vec& u_0){
	// TODO: Legge til kommentar.
	int N_square = N * N;
	arma::cx_vec u_next(N_square);

	u_next(0) = (b(0) + r * (u_0(1) + u_0(N))) * a_diag_inv(0);

	// The biggest loop is here. Parallelising this.
	//#pragma omp parallel
	//{
	//#pragma omp for
	for(int k = N; k < N_square - N; k++){
		u_next(k) = (b(k) + r * (u_0(k - N) + u_0(k - 1) + u_0(k + 1) + u_0(k + N))) * a_diag_inv(k);
	}
	//}

	int help_int;
	for(int k = 1; k < N; k++){
		u_next(k) = (b(k) + r * (u_0(k - 1) + u_0(k + 1) + u_0(k + N))) * a_diag_inv(k);
		help_int = N_square - 1 - k;	// integer in [N_square - N, N_square - 2]
		u_next(help_int) = (b(help_int) + r * (u_0(help_int - N) + u_0(help_int - 1) + u_0(help_int + 1))) * a_diag_inv(help_int);
		u_next(N * k - 1) -= r * u_0(N * k) * a_diag_inv(N * k - 1);
		u_next(N * k) -= r * u_0(N * k - 1) * a_diag_inv(N * k);
	}
	u_next(N_square - 1) = (b(N_square - 1) + r * (u_0(N_square - 2) + u_0(N_square - 1 - N))) * a_diag_inv(N_square - 1);
	return u_next;
}


void Wave_fnc_system::time_step_jacobi(double eps, int max_it) {
	int N_square = N * N;
	arma::cx_vec b = B_mat_times_u();
	arma::cx_vec u_0 = probability_vec;
	arma::cx_vec u_next = jacobi_iteration(b, u_0);
	// double prob_norm = arma::norm(u_0, 2);	// Using conventional 2-norm. Might use this as denominator to avoid many arma::norm() evaluations.
	int it = 0;
	while(arma::norm(u_0 - u_next, 2) / arma::norm(u_0, 2) > eps && it < max_it){
		it += 1;
		u_0 = u_next;
		u_next = jacobi_iteration(b, u_0);
	}
	if(it == max_it){
		std::cout << "Max iterations reached in Jacobi solver. Max iterations = " << max_it << std::endl;
		throw std::runtime_error("Jacobi method did not converge!");
	}
	// For testing:
	std::cout << "For eps = " << eps << ", time_step_jacobi converged after number of iterations it = " << it << std::endl;
	// If the convergence criterion was met:
	probability_vec = u_next;
}


void Wave_fnc_system::time_evolution_jacobi(int n_steps, double eps, int max_it) {
	for(int i = 0; i < n_steps; i++) {
		time_step_jacobi(eps, max_it);
	}
}


//////////////////////////////////

//////////////////////////////////
// Gauss-Seidel iteration method.


arma::cx_vec Wave_fnc_system::gs_iteration(const arma::cx_vec& b, const arma::cx_vec& u_0) {
	int N_square = N * N;
	arma::cx_vec u_next(N_square);
	u_next(0) = a_diag_inv(0) * (b(0) + r * (u_0(1) + u_0(N)));
	for(int k = 1; k < N; k++) {
		u_next(k) = a_diag_inv(k) * (b(k) + r * (u_next(k - 1) + u_0(k + 1) + u_0(k + N)));
	}
	u_next(N - 1) -= a_diag_inv(N - 1) * r * u_0(N);
	for(int k = N; k < N_square - N; k++) {
		// Need a number which is 0 when k = N * i - 1, 1 otherwise. Call this super_diag_nonzero. bool fits this purpose, as bool * double is defined as we want.
		// Need another number which is 0 when k = N * i, 1 otherwise. Call this sub_diag_nonzero.
		bool super_diag_nonzero = (k + 1) % N;
		bool sub_diag_nonzero = k % N;
		std::complex<double> k_minus_1(u_next(k - 1).real() * sub_diag_nonzero, u_next(k - 1).imag() * sub_diag_nonzero);
		std::complex<double> k_plus_1(u_0(k + 1).real() * super_diag_nonzero, u_0(k + 1).imag() * super_diag_nonzero);
		u_next(k) = a_diag_inv(k) * (b(k) + r * (u_next(k - N) + k_minus_1 + k_plus_1 + u_0(k + N)));
	}
	u_next(N_square - N) = a_diag_inv(N_square - N) * (a_diag_inv(N_square - N) + r * (u_next(N_square - 2 * N) + u_0(N_square - N + 1)));
	for(int k = N_square - N + 1; k < N_square - 1; k++) {
		u_next(k) = a_diag_inv(k) * (b(k) + r * (u_next(k - N) + u_next(k - 1) + u_0(k + 1)));
	}
	u_next(N_square - 1) = a_diag_inv(N_square - 1) * (b(N_square - 1) + r * (u_next(N_square - 1 - N) + u_next(N_square - 2)));
	return u_next;
}


void Wave_fnc_system::time_step_gs(double eps, int max_it) {
	// Gauss-Seidel for solving A u^(n + 1) = B u^n = b.
	// Calculating first B u = b.
	arma::cx_vec b = B_mat_times_u();
	arma::cx_vec u_0 = probability_vec;
	arma::cx_vec u_next = gs_iteration(b, u_0);
	// double prob_norm = arma::norm(u, 2); // Using this in denominator saves many evaluations of 1 / arma::norm().
	int it = 0;
	while(arma::norm(u_next - u_0, 2) / arma::norm(u_0, 2) && it < max_it) {
		it += 1;
		u_0 = u_next;
		u_next = gs_iteration(b, u_0);
	}
	if(it == max_it){
		std::cout << "Max iterations reached in Gauss-Seidel solver. Max iterations = " << max_it << std::endl;
		throw std::runtime_error("Gauss-Seidel method did not converge!");
	}
	// For testing:
	std::cout << "For eps = " << eps << ", time_step_gs converged after number of iterations it = " << it << std::endl;
	// If the convergence criterion was met:
	probability_vec = u_next;
}


void Wave_fnc_system::time_evolution_gs(int n_steps, double eps, int max_it) {
	for(int i = 0; i < n_steps; i++) {
		time_step_gs(eps, max_it);
	}
}

//////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Matrix functionality from Armadillo

arma::cx_vec Wave_fnc_system::jacobi_iteration_mat_mult(const arma::cx_vec& b, const arma::cx_vec& u_0) {
	arma::cx_vec U_plus_L_dot_prob_vec_plus_b = b - A_mat_no_diag() * u_0;	// Note minus sign.
	arma::cx_vec u_next(N * N);
	for(int k = 0; k < N * N; k++) {
		u_next(k) = a_diag_inv(k) * U_plus_L_dot_prob_vec_plus_b(k);
	}
	return u_next;
}


void Wave_fnc_system::time_step_arma() {
	// Solving A u^(n+1) = B u^n for u^(n+1).
	arma::cx_vec b = B_sp * probability_vec;	// Calculating B u^n = b.
	probability_vec = arma::spsolve(A_sp, b);	// Solving A u^(n+1) = b, and assigning the solution to probability_vec.
}


/////////////////////////////////////////////////////////////////////////////////
//  Functionality for constructing various matrices (both of dense and sparse types)

arma::cx_mat Wave_fnc_system::prob_vec_to_matrix() const {
	// Writing probability_vec to its matrix form.
	arma::cx_mat prob_mat(N, N);
	for(int k = 0; k < N * N; k++) {
		// Note: Below we intentionally use the nature of int / int.
		prob_mat(k % N, k / N) = probability_vec(k);
	}
	return prob_mat;
}



arma::sp_cx_mat Wave_fnc_system::B_sp_mat() {
	int N_square = N * N;
	arma::sp_cx_mat B_sp(N_square, N_square);
	// Filling diagonal.
	for(int k = 0; k < N_square; k++) {
		B_sp(k, k) = b_diag(k);
	}
	// Filling sub and superdiagonal
	for(int k = 0; k < N_square - 1; k++){
		B_sp(k, k + 1) = r;
		B_sp(k + 1, k) = r;
	}
	// Removing r from the places where it should.
	for(int k = 0; k < N - 1; k++) {
		B_sp(N + N * k - 1, N + N * k) -= r;
		B_sp(N + N * k, N + N * k - 1) -= r;
	}
	// Filling the sub-N-diagonal and super-N-diagonal:
	for(int k = 0; k < N_square - N; k++) {
		B_sp(k, k + N) = r;
		B_sp(k + N, k) = r;
	}
	return B_sp;
}

arma::sp_cx_mat Wave_fnc_system::A_sp_mat() {
	int N_square = N * N;
	arma::sp_cx_mat A_sp(N_square, N_square);
	// Filling diagonal.
	for(int k = 0; k < N_square; k++) {
		A_sp(k, k) = 1.0 / a_diag_inv(k);
	}
	// Filling sub and superdiagonal
	for(int k = 0; k < N_square - 1; k++){
		A_sp(k, k + 1) = -r;
		A_sp(k + 1, k) = -r;
	}
	// Removing r from the places where it should.
	for(int k = 0; k < N - 1; k++) {
		A_sp(N + N * k - 1, N + N * k) += r;
		A_sp(N + N * k, N + N * k - 1) += r;
	}
	// Filling the sub-N-diagonal and super-N-diagonal:
	for(int k = 0; k < N_square - N; k++) {
		A_sp(k, k + N) = -r;
		A_sp(k + N, k) = -r;
	}
	return A_sp;
}

arma::cx_mat Wave_fnc_system::A_mat() {
	arma::cx_mat A(this->A_sp_mat());
	return A;
}



arma::cx_mat Wave_fnc_system::B_mat() {
	arma::cx_mat B(this->B_sp_mat());
	return B;
}


arma::sp_cx_mat Wave_fnc_system::A_mat_no_diag(){
	// For testing purposes.
	int N_square = N * N;
	arma::sp_cx_mat A_no_diag(N_square, N_square);
	// Filling sub and superdiagonal
	for(int k = 0; k < N_square - 1; k++){
		A_no_diag(k, k + 1) = -r;
		A_no_diag(k + 1, k) = -r;
	}
	// Removing r from the places where it should.
	for(int k = 0; k < N - 1; k++) {
		A_no_diag(N + N * k - 1, N + N * k) += r;
		A_no_diag(N + N * k, N + N * k - 1) += r;
	}
	// Filling the sub-N-diagonal and super-N-diagonal:
	for(int k = 0; k < N_square - N; k++) {
		A_no_diag(k, k + N) = -r;
		A_no_diag(k + N, k) = -r;
	}
	return A_no_diag;
}
