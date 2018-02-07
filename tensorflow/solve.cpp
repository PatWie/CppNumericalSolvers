// 2018, Patrick Wieschollek <mail@patwie.com>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <string>
#include <iostream>
#include <cppoptlib/meta.h>
#include <cppoptlib/problem.h>
#include <cppoptlib/solver/bfgssolver.h>

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;


/**
 * @brief load a previous store model
 * @details https://github.com/PatWie/tensorflow_inference
 *
 * in Python run:
 *
 *    saver = tf.train.Saver(tf.global_variables())
 *    saver.save(sess, './exported/my_model')
 *    tf.train.write_graph(sess.graph, '.', './exported/graph.pb, as_text=False)
 *
 * this relies on a graph which has an operation called `init` responsible to initialize all variables, eg.
 *
 *    sess.run(tf.global_variables_initializer())  # somewhere in the python file
 *
 * @param sess active tensorflow session
 * @param graph_fn path to graph file (eg. "./exported/graph.pb")
 * @param checkpoint_fn path to checkpoint file (eg. "./exported/my_model", optional)
 * @return status of reloading
 */
tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn, std::string checkpoint_fn = "") {
  tensorflow::Status status;

  // see https://github.com/PatWie/tensorflow_inference
  tensorflow::MetaGraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK())
    return status;

  status = sess->Create(graph_def.graph_def());
  if (status != tensorflow::Status::OK())
    return status;

  if (checkpoint_fn != "") {
    tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpoint_fn;

    tensor_dict feed_dict = {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
    status = sess->Run(feed_dict, {}, {graph_def.saver_def().restore_op_name()}, nullptr);
    if (status != tensorflow::Status::OK())
      return status;
  } else {
    status = sess->Run({}, {}, {"init"}, nullptr);
    if (status != tensorflow::Status::OK())
      return status;
  }

  return tensorflow::Status::OK();
}



template<typename T>
class Simple : public cppoptlib::Problem<T> {

  tensorflow::Session *sess;
  tensorflow::SessionOptions options;
  tensorflow::Tensor* x;

 public:
  using typename cppoptlib::Problem<T>::TVector;

  Simple() {
    const std::string graph_fn = "/tmp/my_problem/my_problem.meta";
    const std::string checkpoint_fn = "/tmp/my_problem/my_problem";

    TF_CHECK_OK(tensorflow::NewSession(options, &sess));
    TF_CHECK_OK(LoadModel(sess, graph_fn, checkpoint_fn));

    tensorflow::TensorShape data_shape({1, 2});
    x = new tensorflow::Tensor(tensorflow::DT_FLOAT, data_shape);

  }

  T value(const TVector &x0) {

    x->flat<float>().data()[0] = x0[0];
    x->flat<float>().data()[1] = x0[1];

    tensor_dict feed_dict = {
      { "input", *x },
    };

    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(sess->Run(feed_dict, {"problem_objective"}, {}, &outputs));

    return outputs[0].flat<float>()(0);
  }


  void gradient(const TVector &x0, TVector &grad) {

    x->flat<float>().data()[0] = x0[0];
    x->flat<float>().data()[1] = x0[1];
    tensor_dict feed_dict = {
      { "input", *x },
    };
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(sess->Run(feed_dict, {"problem_gradient"}, {}, &outputs));


    grad[0]  = outputs[0].flat<float>()(0);
    grad[1]  = outputs[0].flat<float>()(1);
  }
};

int main(int argc, char const *argv[]) {

  Simple<float> f;
  Eigen::VectorXf x(2); x << 10, 1;

  cppoptlib::BfgsSolver<Simple<float>> solver;
  solver.minimize(f, x);
  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "x " << x.transpose() << std::endl;
  std::cout << "Solver status: " << solver.status() << std::endl;
  std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;
  return 0;
}


