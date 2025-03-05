#include <Eigen/Core>
#include <iostream>

// ======================================================================
// 1. The Function Interface for Differentiability::First
// ======================================================================
namespace cppoptlib {
namespace function {

// Define differentiability levels.
enum class Differentiability { First, Second };

// Base interface for functions with first‐order derivatives.
// For first‐order functions, we only require a gradient.
template <class TScalar, int TDim, Differentiability DiffLevel> class Function {
public:
  using scalar_t = TScalar;
  using vector_t = Eigen::Matrix<TScalar, TDim, 1>;
  static constexpr int Dim = TDim;

  // Pure virtual operator() returning the function value and (optionally) the
  // gradient. Note: the default argument is provided here.
  virtual scalar_t operator()(const vector_t &x,
                              vector_t *gradient = nullptr) const = 0;

  virtual ~Function() = default;
};

// Convenient alias for first‐order functions with dynamic dimensions.
using Function1st = Function<double, Eigen::Dynamic, Differentiability::First>;

// ======================================================================
// 2. Add‐Expression: Sum of Two Functions
// ======================================================================
// AddFunction implements the sum f(x)=f1(x)+f2(x). If a gradient pointer is
// provided, it returns gradient = grad f1(x) + grad f2(x).
template <typename F1, typename F2>
class AddFunction : public Function<typename F1::scalar_t, F1::Dim,
                                    Differentiability::First> {
public:
  using scalar_t = double;
  using vector_t = Eigen::VectorXd;

  AddFunction(const F1 &f1, const F2 &f2) : f1_(f1), f2_(f2) {}

  // IMPORTANT: Do not repeat the default argument here!
  virtual scalar_t operator()(const vector_t &x,
                              vector_t *gradient) const override {
    vector_t grad1, grad2;
    scalar_t val1 = f1_(x, gradient ? &grad1 : nullptr);
    scalar_t val2 = f2_(x, gradient ? &grad2 : nullptr);
    if (gradient) {
      *gradient = grad1 + grad2;
    }
    return val1 + val2;
  }

private:
  F1 f1_;
  F2 f2_;
};

// Overload operator+ to produce an AddFunction.
template <typename F1, typename F2>
AddFunction<F1, F2> operator+(const F1 &f1, const F2 &f2) {
  return AddFunction<F1, F2>(f1, f2);
}

} // namespace function
} // namespace cppoptlib

// ======================================================================
// 3. Concrete Function 1: LinearFunction
//    f(x) = a^T x + b, with constant gradient = a.
// ======================================================================
class LinearFunction : public cppoptlib::function::Function1st {
public:
  using vector_t = Eigen::VectorXd;
  using scalar_t = double;

  // Constructor: specify the coefficient vector and constant.
  LinearFunction(const vector_t &a, scalar_t b) : a_(a), b_(b) {}

  virtual scalar_t operator()(const vector_t &x,
                              vector_t *gradient = nullptr) const override {
    if (gradient) {
      *gradient = a_;
    }
    return a_.dot(x) + b_;
  }

private:
  vector_t a_;
  scalar_t b_;
};

// ======================================================================
// 4. Concrete Function 2: QuadraticFunction
//    f(x) = 0.5 * x^T Q x + b^T x + c, with gradient = Q x + b.
// ======================================================================
class QuadraticFunction : public cppoptlib::function::Function1st {
public:
  using vector_t = Eigen::VectorXd;
  using scalar_t = double;

  QuadraticFunction(const Eigen::MatrixXd &Q, const vector_t &b, scalar_t c)
      : Q_(Q), b_(b), c_(c) {}

  virtual scalar_t operator()(const vector_t &x,
                              vector_t *gradient = nullptr) const override {
    scalar_t val = 0.5 * x.dot(Q_ * x) + b_.dot(x) + c_;
    if (gradient) {
      *gradient = Q_ * x + b_;
    }
    return val;
  }

private:
  Eigen::MatrixXd Q_;
  vector_t b_;
  scalar_t c_;
};

// ======================================================================
// 5. Main: Demonstrate Adding Two Different Functions f1 + f2
// ======================================================================
int main() {
  using namespace Eigen;
  using namespace cppoptlib::function;

  // Define a LinearFunction in R^2: f1(x) = [1, 2]^T * x + 3.
  VectorXd a(2);
  a << 1.0, 2.0;
  double b = 3.0;
  LinearFunction f1(a, b);

  // Define a QuadraticFunction in R^2: f2(x) = 0.5 * x^T Q x + b^T x + c.
  MatrixXd Q(2, 2);
  Q << 4.0, 0.5, 0.5, 3.0;
  VectorXd bvec(2);
  bvec << 1.0, -1.0;
  double c = 2.0;
  QuadraticFunction f2(Q, bvec, c);

  // Use the overloaded operator+ to create a new function f = f1 + f2.
  auto f = f1 + f2;

  // Evaluate f at a sample point x.
  VectorXd x(2);
  x << 0.5, -1.0;
  VectorXd grad(2);
  double val = f(x, &grad);

  std::cout << "f(x) = " << val << "\n";
  std::cout << "grad(x) = " << grad.transpose() << "\n";

  return 0;
}
