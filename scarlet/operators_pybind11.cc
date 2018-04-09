#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <algorithm>

namespace py = pybind11;

typedef Eigen::Matrix<int, Eigen::Dynamic, 1> EigenVector;

void prox_monotonic(
  // Fast implementation of monotonicity constraint
  py::array_t<double> &X,
  double const &step,
  std::vector<int> const &ref_idx,
  std::vector<int> const &dist_idx,
  double const &thresh
){
  auto x = X.mutable_unchecked<1>();
  // Start at the center of the image and set each pixel to the minimum
  // between itself and its reference pixel (which is closer to the peak)
  for(auto &didx: dist_idx){
    x(didx) = std::min(x(didx), x(ref_idx[didx])*(1-thresh));
  }
}

void prox_weighted_monotonic(
    // Fast implementation of weighted monotonicity constraint
    py::array_t<double> &X,
    double const &step,
    py::array_t<double> &weights,
    std::vector<int> const &offsets,
    std::vector<int> const &dist_idx,
    double const &thresh
){
    auto x = X.mutable_unchecked<1>();
    auto w = weights.mutable_unchecked<2>();

    // Start at the center of the image and set each pixel to the minimum
    // between itself and its reference pixel (which is closer to the peak)
    for(auto &didx: dist_idx){
        double ref_flux = 0;
        for(std::size_t i=0; i<offsets.size(); i++){
            if(w(i,didx)>0){
                int nidx = offsets[i] + didx;
                ref_flux += x(nidx) * w(i, didx);
            }
        }
        x(didx) = std::min(x(didx), ref_flux*(1-thresh));
    }
}

// Apply a filter to an image
template <typename M, typename V>
M apply_filter(
    Eigen::Ref<const M> image,
    Eigen::Ref<const V> values,
    Eigen::Ref<const EigenVector> y_start,
    Eigen::Ref<const EigenVector> y_end,
    Eigen::Ref<const EigenVector> x_start,
    Eigen::Ref<const EigenVector> x_end
){
    M result(image.rows(), image.cols());
    result.setZero(image.rows(), image.cols());
    for(int n=0; n<values.size(); n++){
        int rows = image.rows()-y_start(n)-y_end(n);
        int cols = image.cols()-x_start(n)-x_end(n);
        result.block(y_start(n), x_start(n), rows, cols) +=
            values(n) * image.block(y_end(n), x_end(n), rows, cols);
    }
    return result;
}

PYBIND11_PLUGIN(operators_pybind11)
{
  py::module mod("operators_pybind11", "Fast proximal operators");
  mod.def("prox_monotonic", &prox_monotonic, "Monotonic Proximal Operator");
  mod.def("prox_weighted_monotonic", &prox_weighted_monotonic, "Weighted Monotonic Proximal Operator");
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorF;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixD;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorD;

  mod.def("apply_filter", &apply_filter<MatrixF, VectorF>);
  mod.def("apply_filter", &apply_filter<MatrixD, VectorD>);
  return mod.ptr();
}
