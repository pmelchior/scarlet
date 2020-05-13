#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <algorithm>

namespace py = pybind11;

typedef Eigen::Array<int, Eigen::Dynamic, 1> IndexVector;

template <typename T, typename M, typename V>
void prox_weighted_monotonic(
    // Fast implementation of weighted monotonicity constraint
    Eigen::Ref<V> flat_img,
    Eigen::Ref<const M> weights,
    Eigen::Ref<const IndexVector> offsets,
    Eigen::Ref<const IndexVector> dist_idx,
    T const &min_gradient
){
    // Start at the center of the image and set each pixel to the minimum
    // between itself and its reference pixel (which is closer to the peak)
    for(int d=0; d<dist_idx.size(); d++){
        int didx = dist_idx(d);
        T ref_flux = 0;
        for(int i=0; i<offsets.size(); i++){
            if(weights(i,didx)>0){
                int nidx = offsets[i] + didx;
                ref_flux += flat_img(nidx) * weights(i, didx);
            }
        }
        flat_img(didx) = std::min(flat_img(didx), ref_flux*(1-min_gradient));
    }
}

// Apply a filter to an image
template <typename M, typename V>
void apply_filter(
    Eigen::Ref<const M> image,
    Eigen::Ref<const V> values,
    Eigen::Ref<const IndexVector> y_start,
    Eigen::Ref<const IndexVector> y_end,
    Eigen::Ref<const IndexVector> x_start,
    Eigen::Ref<const IndexVector> x_end,
    Eigen::Ref<M, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> result
){
    result.setZero();
    for(int n=0; n<values.size(); n++){
        int rows = image.rows()-y_start(n)-y_end(n);
        int cols = image.cols()-x_start(n)-x_end(n);
        result.block(y_start(n), x_start(n), rows, cols) +=
            values(n) * image.block(y_end(n), x_end(n), rows, cols);
    }
}

PYBIND11_PLUGIN(operators_pybind11)
{
  py::module mod("operators_pybind11", "Fast proximal operators");

  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixF;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorF;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixD;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorD;

  mod.def("prox_weighted_monotonic", &prox_weighted_monotonic<float, MatrixF, VectorF>,
          "Weighted Monotonic Proximal Operator");
  mod.def("prox_weighted_monotonic", &prox_weighted_monotonic<double, MatrixD, VectorD>,
          "Weighted Monotonic Proximal Operator");

  mod.def("apply_filter", &apply_filter<MatrixF, VectorF>, "Apply a filter to a 2D image");
  mod.def("apply_filter", &apply_filter<MatrixD, VectorD>, "Apply a filter to a 2D image");

  return mod.ptr();
}
