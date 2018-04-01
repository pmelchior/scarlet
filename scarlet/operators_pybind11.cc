#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>

namespace py = pybind11;

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
    double ref_flux;

    // Start at the center of the image and set each pixel to the minimum
    // between itself and its reference pixel (which is closer to the peak)
    for(auto &didx: dist_idx){
        ref_flux = 0;
        for(std::size_t i=0; i<offsets.size(); i++){
            if(w(i,didx)>0){
                int nidx = offsets[i] + didx;
                ref_flux += x(nidx) * w(i, didx);
            }
        }
        x(didx) = std::min(x(didx), ref_flux*(1-thresh));
    }
}

// This function is currently unused but might be useful to implement
// and speed up PSF covolution
void apply_filter(
    py::array_t<double> &X,
    py::array_t<double> &weights,
    std::vector<std::vector<std::vector<ssize_t>>> &slices,
    std::vector<std::vector<std::vector<ssize_t>>> &inv_slices,
    py::array_t<double> &result
){
    assert((slices.size()==inv_slices.size()) && (weights.shape(0))==slices.size());
    auto x = X.mutable_unchecked<2>();
    auto w = weights.mutable_unchecked<1>();
    auto r = result.mutable_unchecked<2>();

    for(ssize_t n=0; n<w.shape(0); n++){
        ssize_t y_min = slices[n][0][0];
        ssize_t dy = slices[n][0][1] - y_min;
        ssize_t x_min = slices[n][1][0];
        ssize_t dx = slices[n][1][1] - x_min;
        ssize_t inv_y_min = inv_slices[n][0][0];
        ssize_t inv_x_min = inv_slices[n][1][0];
        for(ssize_t i=0; i<dy; i++){
            for(ssize_t j=0; j<dx; j++){
                r(i+y_min, i+x_min) += w(i) * x(i+inv_y_min, i+inv_x_min);
            }
        }
    }
}

PYBIND11_PLUGIN(operators_pybind11)
{
  py::module mod("operators_pybind11", "Fast proximal operators");
  mod.def("prox_monotonic", &prox_monotonic, "Monotonic Proximal Operator");
  mod.def("prox_weighted_monotonic", &prox_weighted_monotonic, "Weighted Monotonic Proximal Operator");
  mod.def("apply_filter", &apply_filter, "Apply a filter to a 2D array");
  return mod.ptr();
}
