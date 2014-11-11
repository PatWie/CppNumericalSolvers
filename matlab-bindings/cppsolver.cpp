// Patrick Wieschollek
// for compiling download eigen and call "mex -I./eigen cppnumerical.cpp;"
#include <vector>
#include <Eigen/Eigen>
#include <iostream>
#include <functional>
#include "../../src/LbfgsSolver.h"
#include "../../src/BfgsSolver.h"
#include "../../src/GradientDescentSolver.h"
#include "mex.h"

/* usage: cppsolver(x0,@objective,[args]) 
args = 'gradient', @gradient
       'solver', ["stochgrad"|"bfgs"|"l-bfgs"|"l-bfgs-b"]
*/

char *objective_name;
char *gradient_name;
bool has_gradient;
char error_msg[200];

Eigen::VectorXd solution,gradient;

enum solver_type{GRADIENTDESC,NEWTON,BFGS,LBFGS,LBFGSB};
solver_type selected_solver;

void mexFunction(int outLen, mxArray* outArr[], int inLen, const mxArray* inArr[]) {
    has_gradient = false;
    bool skip_gradient_check = false;
    solution = -1*Eigen::VectorXd::Ones(1);
    selected_solver = solver_type::LBFGS;
    // for error messages
    
    if (inLen < 2){
    mexErrMsgIdAndTxt("MATLAB:cppsolver", "this function need at leat one parameter");
    }



    // PARSING PARAMETERS

    // initial solution
    size_t in_rows = mxGetM(inArr[0]);
    size_t in_cols = mxGetN(inArr[0]);
    if(in_cols > 1 || in_rows == 0){
        sprintf(error_msg, "The first argument has to be the inital guess x0 (format: n x 1), but the input format is %d x %d",in_rows,in_cols);
        mexErrMsgIdAndTxt("MATLAB:cppsolver", error_msg);
    }
    Eigen::Map<Eigen::VectorXd> initial_guess = Eigen::Map<Eigen::VectorXd>(mxGetPr(inArr[0]), mxGetM(inArr[0])*mxGetN(inArr[0])); 
    solution = initial_guess;

    // function handle "@objective"
    if (mxGetClassID(inArr[1]) != mxFUNCTION_CLASS) {
        mexErrMsgIdAndTxt("MATLAB:cppsolver", "the second arguments has to be the handle of the function (@objective)");
    }
    mxArray *objective_ans,*objective_param[1];
    // get name of objective
    objective_param[0] = const_cast<mxArray *>( inArr[1] );
    mexCallMATLAB(1, &objective_ans, 1,objective_param, "char") ;
    objective_name =   mxArrayToString(objective_ans);

    mexPrintf("Found objective function: %s\n",objective_name);


    if(inLen > 2){
        // there are some parameters
        if((inLen%2) != 0){
            mexErrMsgIdAndTxt("MATLAB:cppsolver", "optional arguments have to be passed by 'key','value'.");
        }
        for(int arg=2;arg<inLen;arg+=2){
            if(!mxIsChar( inArr[arg])){
                mexErrMsgIdAndTxt("MATLAB:cppsolver", "optional argument keys have to be strings");
            }
            char *key_str = mxArrayToString(inArr[arg]);
            printf("parsing key: %s\n",key_str);

            if(strcmp(key_str,"gradient")== 0){
                // extract gradient name
                if (mxGetClassID(inArr[arg+1]) != mxFUNCTION_CLASS) {
                    mexErrMsgIdAndTxt("MATLAB:cppsolver", "the argument following 'gradient' has to a function handle (@gradient)");
                }
                objective_param[0] = const_cast<mxArray *>( inArr[arg+1] );
                mexCallMATLAB(1, &objective_ans, 1,objective_param, "char") ;
                gradient_name =   mxArrayToString(objective_ans);
                has_gradient = true;
                mexPrintf("Found gradient function: %s\n",gradient_name);
            }else{

                if(strcmp(key_str,"solver")== 0){
                  if(!mxIsChar( inArr[arg+1])){
                      mexErrMsgIdAndTxt("MATLAB:cppsolver", "solver name has to be a string");
                  }
                  char *solver_str = mxArrayToString(inArr[arg+1]);
                  
                  if(strcmp(solver_str,"gradientdescent")== 0){
                    selected_solver = GRADIENTDESC;
                  }else if(strcmp(solver_str,"bfgs")== 0){
                    selected_solver = BFGS;
                  }else if(strcmp(solver_str,"l-bfgs")== 0){
                    selected_solver = LBFGS;
                  }else if(strcmp(solver_str,"l-bfgs-b")== 0){
                    selected_solver = LBFGSB;
                  }else if(strcmp(solver_str,"newton")== 0){
                    selected_solver = NEWTON;
                  }else{
                    sprintf(error_msg, "unknown solver %s",solver_str);
                    mexErrMsgIdAndTxt("MATLAB:cppsolver", error_msg); 

                  }


                }else{

                  if(strcmp(key_str,"skip_gradient_check")== 0){
                    if(!mxIsChar( inArr[arg+1])){
                      mexErrMsgIdAndTxt("MATLAB:cppsolver", "the value of the key 'skip_gradient_check' has to be a string");
                     }
                     char *txt = mxArrayToString(inArr[arg+1]);
                     if(strcmp(txt,"true")== 0){
                    skip_gradient_check = true;
                  }
                  }else{
                    sprintf(error_msg, "unknown argument %s",key_str);
                    mexErrMsgIdAndTxt("MATLAB:cppsolver", error_msg);  
                  }

                    
                }

            
                     
            }
        }
    }

    // solve
    auto objective_function = [](const Eigen::VectorXd & x) -> double
    {
        mxArray *objective_ans,*objective_param[1];
        objective_param[0] = mxCreateDoubleMatrix(x.rows(), x.cols(), mxREAL);
        const double* constVariablePtr = &x(0);
        memcpy(mxGetPr(objective_param[0]), constVariablePtr, mxGetM(objective_param[0]) * mxGetN(objective_param[0]) * sizeof(*constVariablePtr)); 
        mexCallMATLAB(1, &objective_ans, 1,objective_param, objective_name) ;
        return mxGetScalar(objective_ans);
    };

    // create derivative of function
    auto gradient_function = [&](const Eigen::VectorXd x, Eigen::VectorXd & grad) -> void
    {
        mxArray *objective_ans,*objective_param[1];
        objective_param[0] = mxCreateDoubleMatrix(x.rows(), x.cols(), mxREAL);
        const double* constVariablePtr = &x(0);
        memcpy(mxGetPr(objective_param[0]), constVariablePtr, mxGetM(objective_param[0]) * mxGetN(objective_param[0]) * sizeof(*constVariablePtr)); 

        mexCallMATLAB(1, &objective_ans, 1,objective_param, gradient_name) ;
        size_t r = mxGetM(objective_ans);
        size_t c = mxGetN(objective_ans);
        if((in_rows != r) || (in_cols != c)){
          sprintf(error_msg, "Wrong format of gradient! The correct format is %d x %d, but %d x %d was given",in_rows,in_cols,r,c);
          mexErrMsgIdAndTxt("MATLAB:cppsolver", error_msg);
        }

        grad = Eigen::Map<Eigen::VectorXd>(mxGetPr(objective_ans), mxGetM(objective_ans)*mxGetN(objective_ans)); 
        //mexPrintf("gradient: %f %f\n",gradient[0],gradient[1]);

    };

    // check gradient
    if(has_gradient && !skip_gradient_check){

      Eigen::VectorXd dx = solution;
      gradient_function(solution,dx);
      if(!pwie::checkGradient(objective_function, solution, dx))
      {
          mexErrMsgIdAndTxt("MATLAB:cppsolver:gradient_check", "your gradient seems to be not correct! You can skip this test by using the arguments \"'skip_gradient_check','true'\"");
      }
    }

    gradient = solution;

    switch(selected_solver){
      case LBFGS:
      {
        pwie::LbfgsSolver g;
        if(has_gradient){
          g.solve(solution, objective_function,gradient_function);
        }else{
          g.solve(solution, objective_function);
        }
      }
      break;
      case BFGS:
      {
        pwie::BfgsSolver g;
        if(has_gradient){
          g.solve(solution, objective_function,gradient_function);
        }else{
          g.solve(solution, objective_function);
        }
      }
      break;
      case GRADIENTDESC:
      {
        pwie::GradientDescentSolver g;
        if(has_gradient){
          g.solve(solution, objective_function,gradient_function);
        }else{
          g.solve(solution, objective_function);
        }
      }
      break;
      default:
        mexErrMsgIdAndTxt("MATLAB:cppsolver:not_implemented", "your select solver has currently no matlab binding");
        break;
    }

    

    // prepare solution
    outArr[0] = mxCreateDoubleMatrix(solution.rows(), solution.cols(), mxREAL);
    double* constVariablePtr = &solution(0);
    memcpy(mxGetPr(outArr[0]), constVariablePtr, mxGetM(outArr[0]) * mxGetN(outArr[0]) * sizeof(*constVariablePtr)); 

}

