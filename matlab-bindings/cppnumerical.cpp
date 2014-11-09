// Patrick Wieschollek
// for compiling download eigen and call "mex -I./eigen cppnumerical.cpp;"
#include <vector>
#include <Eigen/Eigen>
#include <iostream>
#include "mex.h"


void mexFunction(int outLen, mxArray* outArr[], int inLen, const mxArray* inArr[]) {
    // just testing cppnumerical(guess,@objective)
    if (inLen < 2)
		mexErrMsgIdAndTxt("MATLAB:cppsolver", "need at leat two parameter");

    // initial guess - dimensions
    size_t in_rows = mxGetM(inArr[0]);
    size_t in_cols = mxGetN(inArr[0]);

    if(in_cols > 1)
        mexErrMsgIdAndTxt("MATLAB:cppsolver", "the inital guess has to be a vector");

    Eigen::Map<Eigen::VectorXd> initial_guess = Eigen::Map<Eigen::VectorXd>(mxGetPr(inArr[0]), mxGetM(inArr[0])*mxGetN(inArr[0])); 
    Eigen::VectorXd solution = initial_guess;

    // function handle
    if (mxGetClassID(inArr[1]) != mxFUNCTION_CLASS) {
        mexErrMsgIdAndTxt("MATLAB:cppsolver", "name of objective function is not a string");
    }

    mxArray *objective_ans,*objective_param[1];
    // get name of objective
    objective_param[0] = const_cast<mxArray *>( inArr[1] );
    mexCallMATLAB(1, &objective_ans, 1,objective_param, "char") ;
    char *objective_name =   mxArrayToString(objective_ans);
    //mexPrintf("function name is %s\n",objective_name);
    // prepare value for objective
    objective_param[0] = const_cast<mxArray *>( inArr[0] );
    // call objective
    mexCallMATLAB(1, &objective_ans, 1,objective_param, objective_name) ;
    // back to eigen
    Eigen::VectorXd ans = Eigen::Map<Eigen::VectorXd>(mxGetPr(objective_ans), in_rows* in_cols); 

    // prepare output
    outArr[0] = mxCreateDoubleMatrix(solution.rows(), solution.cols(), mxREAL);
    
    double* constVariablePtr = &ans(0);
    memcpy(mxGetPr(outArr[0]), constVariablePtr, mxGetM(outArr[0]) * mxGetN(outArr[0]) * sizeof(*constVariablePtr)); 



}