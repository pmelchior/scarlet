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
    std::vector<int> const &dist_idx
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
        x(didx) = std::min(x(didx), ref_flux);
    }
}

PYBIND11_PLUGIN(operators_pybind11)
{
  py::module mod("operators_pybind11", "Fast proximal operators");
  mod.def("prox_monotonic", &prox_monotonic, "Monotonic Proximal Operator");
  mod.def("prox_weighted_monotonic", &prox_weighted_monotonic, "Weighted Monotonic Proximal Operator");
  return mod.ptr();
}
