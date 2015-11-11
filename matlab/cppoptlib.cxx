// disable Testing Macros from G-test
#define MATLAB

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "mex.h"
#include "../include/cppoptlib/problem.h"
#include "../include/cppoptlib/meta.h"
#include "../include/cppoptlib/solver/gradientdescentsolver.h"
#include "../include/cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "../include/cppoptlib/solver/newtondescentsolver.h"
#include "../include/cppoptlib/solver/bfgssolver.h"
#include "../include/cppoptlib/solver/lbfgssolver.h"
#include "../include/cppoptlib/solver/lbfgsbsolver.h"

using namespace cppoptlib;

char *nameObjectiveFunction;
char *nameGradientFunction;  bool hasGradient = false;
char *nameHessianFunction;   bool hasHessian  = false;

size_t in_rows;
size_t in_cols;

char error_msg[200];

template<typename T>
class MATLABobjective : public Problem<T> {
 public:
  T value(const Vector<T> &x) {
    mxArray * objective_ans, *objective_param[1];
    objective_param[0] = mxCreateDoubleMatrix(x.rows(), x.cols(), mxREAL);
    const T *constVariablePtr = &x(0);
    memcpy(mxGetPr(objective_param[0]), constVariablePtr, mxGetM(objective_param[0]) * mxGetN(objective_param[0]) * sizeof(*constVariablePtr));
    mexCallMATLAB(1, &objective_ans, 1, objective_param, nameObjectiveFunction) ;
    return mxGetScalar(objective_ans);
  }

  void gradient(const Vector<T> &x, Vector<T> &grad) {
    if (hasGradient) {
      mxArray * objective_ans, *objective_param[1];
      objective_param[0] = mxCreateDoubleMatrix(x.rows(), x.cols(), mxREAL);
      const double *constVariablePtr = &x(0);
      memcpy(mxGetPr(objective_param[0]), constVariablePtr, mxGetM(objective_param[0]) * mxGetN(objective_param[0]) * sizeof(*constVariablePtr));
      mexCallMATLAB(1, &objective_ans, 1, objective_param, nameGradientFunction) ;
      size_t r = mxGetM(objective_ans);
      size_t c = mxGetN(objective_ans);
      if ((in_rows != r) || (in_cols != c)) {
        sprintf(error_msg, "Wrong format of gradient! The correct format is %zu x %zu, but %zu x %zu was given", in_rows, in_cols, r, c);
        mexErrMsgIdAndTxt("MATLAB:cppoptlib", error_msg);
      }

      grad = Eigen::Map<Eigen::VectorXd>(mxGetPr(objective_ans), mxGetM(objective_ans) );

    } else {
      this->finiteGradient(x, grad);
    }
  }

  void hessian(const Vector<T> &x, Matrix<T> & hessian) {
    if (hasHessian) {

      mxArray * objective_ans, *objective_param[1];
      objective_param[0] = mxCreateDoubleMatrix(x.rows(), x.cols(), mxREAL);
      const double *constVariablePtr = &x(0);
      memcpy(mxGetPr(objective_param[0]), constVariablePtr, mxGetM(objective_param[0]) * mxGetN(objective_param[0]) * sizeof(*constVariablePtr));
      mexCallMATLAB(1, &objective_ans, 1, objective_param, nameHessianFunction) ;
      size_t r = mxGetM(objective_ans);
      size_t c = mxGetN(objective_ans);
      if ((in_rows != r) || (in_rows != c) || (c != r)) {
        sprintf(error_msg, "Wrong format of hessian! The correct format is %zu x %zu, but %zu x %zu was given", in_rows, in_rows, r, c);
        mexErrMsgIdAndTxt("MATLAB:cppoptlib", error_msg);
      }

      hessian = Eigen::Map<Eigen::MatrixXd>(mxGetPr(objective_ans), mxGetM(objective_ans) , mxGetN(objective_ans));

    } else {
      this->finiteHessian(x, hessian);
    }
  }


};


void mexFunction(int outLen, mxArray *outArr[], int inLen, const mxArray *inArr[]) {

    MATLABobjective<double> f;

  // check parameters
  if (inLen < 2) {
    mexErrMsgIdAndTxt("MATLAB:cppoptlib", "this function need at leat two parameters");
  }

  if (mxGetClassID(inArr[0]) != mxDOUBLE_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:cppoptlib", "the first arguments needs to be an array of type double");
  }

  // check starting point
  // ----------------------------------------------------------
  in_rows = mxGetM(inArr[0]);
  in_cols = mxGetN(inArr[0]);
  if (in_cols > 1 || in_rows == 0) {
    sprintf(error_msg, "The first argument has to be the inital guess x0 (format: n x 1), but the input format is %zu x %zu", in_rows, in_cols);
    mexErrMsgIdAndTxt("MATLAB:cppoptlib", error_msg);
  }
  Eigen::Map<Eigen::VectorXd> solution = Eigen::Map<Eigen::VectorXd>(mxGetPr(inArr[0]), mxGetM(inArr[0]) * mxGetN(inArr[0]));

  Vector<double> x = solution.eval();
  // check objective function
  // ----------------------------------------------------------
  if (mxGetClassID(inArr[1]) != mxFUNCTION_CLASS) {
    mexErrMsgIdAndTxt("MATLAB:cppoptlib", "the second arguments has to be the handle of the function (@objective)");
  }

  // get name of objective
  // ----------------------------------------------------------
  mxArray *objective_ans, *objective_param[1];
  objective_param[0] = const_cast<mxArray *>( inArr[1] );
  mexCallMATLAB(1, &objective_ans, 1, objective_param, "char") ;
  nameObjectiveFunction =   mxArrayToString(objective_ans);

  // parse remaining arguments
  // ----------------------------------------------------------
  enum solver_type {GRADIENTDESCENT, NEWTON, BFGS, LBFGS, LBFGSB, CONJUGATEDGRADIENTDESCENT};
  solver_type selected_solver = BFGS;

  if (inLen > 2) {
    // there are some parameters
    if ((inLen % 2) != 0) {
      mexErrMsgIdAndTxt("MATLAB:cppoptlib", "optional arguments have to be passed by 'key','value'.");
    }
    for (int arg = 2; arg < inLen; arg += 2) {
      if (!mxIsChar( inArr[arg])) {
        mexErrMsgIdAndTxt("MATLAB:cppoptlib", "optional argument keys have to be strings");
      }
      char *key_str = mxArrayToString(inArr[arg]);

      if (strcmp(key_str, "gradient") == 0) {
        if (mxGetClassID(inArr[arg + 1]) != mxFUNCTION_CLASS) {
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", "the argument following 'gradient' has to be a function handle (@gradient)");
        }
        objective_param[0] = const_cast<mxArray *>( inArr[arg + 1] );
        mexCallMATLAB(1, &objective_ans, 1, objective_param, "char") ;
        nameGradientFunction =   mxArrayToString(objective_ans);
        hasGradient = true;
      }
      if (strcmp(key_str, "hessian") == 0) {
        if (mxGetClassID(inArr[arg + 1]) != mxFUNCTION_CLASS) {
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", "the argument following 'hessian' has to be a function handle (@hessian)");
        }
        objective_param[0] = const_cast<mxArray *>( inArr[arg + 1] );
        mexCallMATLAB(1, &objective_ans, 1, objective_param, "char") ;
        nameHessianFunction =   mxArrayToString(objective_ans);
        hasHessian = true;
      }
      if (strcmp(key_str, "solver") == 0) {
        if (!mxIsChar( inArr[arg + 1])) {
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", "solver name has to be a string");
        }
        char *solver_str = mxArrayToString(inArr[arg + 1]);
        if (strcmp(solver_str, "gradientdescent") == 0) {
          selected_solver = GRADIENTDESCENT;
        } else if (strcmp(solver_str, "cg") == 0) {
          selected_solver = CONJUGATEDGRADIENTDESCENT;
        } else if (strcmp(solver_str, "bfgs") == 0) {
          selected_solver = BFGS;
        } else if (strcmp(solver_str, "l-bfgs") == 0) {
          selected_solver = LBFGS;
        } else if (strcmp(solver_str, "l-bfgs-b") == 0) {
          selected_solver = LBFGSB;
        } else if (strcmp(solver_str, "newton") == 0) {
          selected_solver = NEWTON;
        } else {
          sprintf(error_msg, "unknown solver %s", solver_str);
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", error_msg);
        }
      }
      if (strcmp(key_str, "lb") == 0) {
        if (mxGetClassID(inArr[arg + 1]) != mxDOUBLE_CLASS) {
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", "the argument following 'lb' has to be an array of type double");
        }
        size_t lbr = mxGetM(inArr[arg + 1]);
        size_t lbc = mxGetN(inArr[arg + 1]);

        if ((in_cols != lbc) || (in_rows != lbr)) {
          sprintf(error_msg, "expected lowerBound argument format is (format: %zu x 1), but the input format is %zu x %zu", in_rows, lbr, lbc);
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", error_msg);
        }

        Eigen::Map<Eigen::VectorXd> tmp = Eigen::Map<Eigen::VectorXd>(mxGetPr(inArr[arg + 1]), mxGetM(inArr[arg + 1]) * mxGetN(inArr[arg + 1]));
        Vector<double> t=tmp;
        f.setLowerBound(t);

      }
      if (strcmp(key_str, "ub") == 0) {
        if (mxGetClassID(inArr[arg + 1]) != mxDOUBLE_CLASS) {
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", "the argument following 'lb' has to be an array of type double");
        }
        size_t lbr = mxGetM(inArr[arg + 1]);
        size_t lbc = mxGetN(inArr[arg + 1]);

        if ((in_cols != lbc) || (in_rows != lbr)) {
          sprintf(error_msg, "expected lowerBound argument format is (format: %zu x 1), but the input format is %zu x %zu", in_rows, lbr, lbc);
          mexErrMsgIdAndTxt("MATLAB:cppoptlib", error_msg);
        }

        Eigen::Map<Eigen::VectorXd> tmp = Eigen::Map<Eigen::VectorXd>(mxGetPr(inArr[arg + 1]), mxGetM(inArr[arg + 1]) * mxGetN(inArr[arg + 1]));
        Vector<double> t=tmp;
        f.setUpperBound(t);

      }
    }
  }

  // solve
  // ----------------------------------------------------------



  switch (selected_solver) {
  case LBFGS: {
    LbfgsSolver<double> solver;
    solver.minimize(f, x);
  }
  break;
  case LBFGSB: {
    LbfgsbSolver<double> solver;
    solver.minimize(f, x);
  }
  break;
  case BFGS: {
    BfgsSolver<double> solver;
    solver.minimize(f, x);
  }
  break;
  case GRADIENTDESCENT: {
    GradientDescentSolver<double> solver;
    solver.minimize(f, x);
  }
  break;
  case CONJUGATEDGRADIENTDESCENT: {
    ConjugatedGradientDescentSolver<double> solver;
    solver.minimize(f, x);
  }
  break;
  case NEWTON: {
    NewtonDescentSolver<double> solver;
    solver.minimize(f, x);
  }
  break;
  default:
    mexErrMsgIdAndTxt("MATLAB:cppoptlib:not_implemented", "Your select solver has currently no matlab binding. Oops.");
    break;
  }

  // prepare solution
  outArr[0] = mxCreateDoubleScalar(f(x));
  if (outLen > 1) {
    outArr[1] = mxCreateDoubleMatrix(x.rows(), x.cols(), mxREAL);
    double *constVariablePtr = &x(0);
    memcpy(mxGetPr(outArr[1]), constVariablePtr, mxGetM(outArr[1]) * mxGetN(outArr[1]) * sizeof(*constVariablePtr));
  }



}