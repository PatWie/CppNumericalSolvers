#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/boundedproblem.h"
#include "../../include/cppoptlib/solver/lbfgsbsolver.h"

// to use CppNumericalSolvers just use the namespace "cppoptlib"
namespace cppoptlib {

// we will solve <b,b> s.t. b>=20 and b <= 40
template<typename T>
class NonNegativeLeastSquares : public BoundedProblem<T> {
  public:
    using Superclass = BoundedProblem<T>;
    using typename Superclass::TVector;
    using TMatrix = typename Superclass::THessian;

  public:
    NonNegativeLeastSquares(int dim) :
        Superclass(dim) {}

    T value(const TVector &beta) {
        return beta.dot(beta);
    }

    void gradient(const TVector &beta, TVector &grad) {
        grad = beta * 2;
    }
};

}
int main(int argc, char const *argv[]) {

    const size_t DIM = 4;
    typedef double T;
    typedef cppoptlib::NonNegativeLeastSquares<T> TNNLS;
    typedef typename TNNLS::TVector TVector;

    // perform non-negative least squares
    TNNLS f(4);
    f.setLowerBound(TVector::Ones(DIM) * 20);
    f.setUpperBound(TVector::Ones(DIM) * 40);
    // create initial guess
    TVector beta = TVector::Ones(DIM) * 21;
    cppoptlib::LbfgsbSolver<TNNLS> solver;
    solver.minimize(f, beta);
    std::cout << "final b = " << beta.transpose() << "\tloss:" << f(beta) << std::endl;

    return 0;
}
