#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <math.h>
#include <algorithm>

namespace py = pybind11;
using namespace pybind11::literals;

typedef Eigen::Array<int, 4, 1> Bounds;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixB;


// Create a boolean mask `footprint` for all of the pixels that are connected to the pixel
// located at `i,j` and create the bounding box for the `footprint` in `image`.
template <typename M>
void get_connected_pixels(
    const int i,
    const int j,
    Eigen::Ref<const M> image,
    Eigen::Ref<MatrixB, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> unchecked,
    Eigen::Ref<MatrixB, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> footprint,
    Eigen::Ref<Bounds, 0, Eigen::Stride<4, 1>> bounds,
    const double thresh=0
){
    if(not unchecked(i,j)){
        return;
    }
    unchecked(i,j) = false;

    if(image(i,j) > thresh){
        footprint(i,j) = true;

        if(i < bounds[0]){
            bounds[0] = i;
        } else if(i > bounds[1]){
            bounds[1] = i;
        }
        if(j < bounds[2]){
            bounds[2] = j;
        } else if(j > bounds[3]){
            bounds[3] = j;
        }

        if(i > 0){
            get_connected_pixels(i-1, j, image, unchecked, footprint, bounds, thresh);
        }
        if(i < image.rows()-1){
            get_connected_pixels(i+1, j, image, unchecked, footprint, bounds, thresh);
        }
        if(j > 0){
            get_connected_pixels(i, j-1, image, unchecked, footprint, bounds, thresh);
        }
        if(j < image.cols()-1){
            get_connected_pixels(i, j+1, image, unchecked, footprint, bounds, thresh);
        }
    }
}


// Get a list of peaks found in an image.
// To make ut easier to cull peaks that are too close together
// and ensure that every footprint has at least one peak,
// this algorithm is meant to be run on a single footprint
// created by `get_connected_pixels`.
template <typename M>
std::vector<std::vector<int>> get_peaks(
    //Eigen::Ref<const M> image,
    M& image,
    const double min_separation,
    const int y0,
    const int x0
){
    const int height = image.rows();
    const int width = image.cols();

    std::vector<std::vector<int>> peaks;

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(i > 0 && image(i, j) <= image(i-1, j)){
                continue;
            }
            if(i < height-1 && image(i,j) <= image(i+1, j)){
                continue;
            }
            if(j > 0 && image(i, j) <= image(i, j-1)){
                continue;
            }
            if(j < width-1 && image(i,j) <= image(i, j+1)){
                continue;
            }
            std::vector<int> peak{i+y0, j+x0};
            peaks.push_back(peak);
        }
    }

    assert(peaks.size() > 0);

    // Remove peaks within min_separation
    double min_separation2 = min_separation * min_separation;
    auto p1 = peaks.begin();
    while (p1->size() > 1 && p1 != std::prev(peaks.end())){
        auto p2 = std::next(p1);
        while (p2 != peaks.end()){
            double dy = p1->at(0)-p2->at(0);
            double dx = p1->at(1)-p2->at(1);
            double separation2 = dy*dy + dx*dx;
            if(separation2 < min_separation2){
                if(image(p2->at(0)-y0, p2->at(1)-x0) > image(p1->at(0)-y0, p1->at(1)-x0)){
                    p1 = peaks.erase(p1);
                    break;
                } else {
                    p2 = peaks.erase(p2);
                }
            } else {
                ++p2;
            }
        }
        ++p1;
    }

    assert(peaks.size() > 0);

    return peaks;
}


// A detected footprint
class Footprint {
public:
    Footprint(MatrixB footprint, std::vector<std::vector<int>> peaks, Bounds bounds){
        _footprint = footprint;
        _peaks = peaks;
        _bounds = bounds;
    }

    MatrixB getFootprint(){
        return _footprint;
    }

    std::vector<std::vector<int>> getPeaks(){
        return _peaks;
    }

    Bounds getBounds(){
        return _bounds;
    }

private:
    MatrixB _footprint;
    std::vector<std::vector<int>> _peaks;
    Bounds _bounds;
};


template <typename M>
void maskImage(
    Eigen::Ref<M, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> image,
    Eigen::Ref<MatrixB, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> footprint
){
    const int height = image.rows();
    const int width = image.cols();

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(!footprint(i,j)){
                image(i,j) = 0;
            }
        }
    }
}


template <typename M, typename P>
std::vector<Footprint> get_footprints(
    Eigen::Ref<const M> image,
    const double min_separation,
    const int min_area,
    const int thresh
){
    const int height = image.rows();
    const int width = image.cols();

    std::vector<Footprint> footprints;
    MatrixB unchecked = MatrixB::Ones(height, width);
    MatrixB footprint = MatrixB::Zero(height, width);

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            Bounds bounds; bounds << i, i, j, j;
            get_connected_pixels(i, j, image, unchecked, footprint, bounds, thresh);
            int subHeight = bounds[1]-bounds[0]+1;
            int subWidth = bounds[3]-bounds[2]+1;
            if(subHeight * subWidth > min_area){
                MatrixB subFootprint = footprint.block(bounds[0], bounds[2], subHeight, subWidth);
                int area = subFootprint.count();
                if(area >= min_area){
                    M patch = image.block(bounds[0], bounds[2], subHeight, subWidth);
                    maskImage<M>(patch, subFootprint);
                    std::vector<std::vector<int>> _peaks = get_peaks(
                        patch,
                        min_separation,
                        bounds[0],
                        bounds[2]
                    );
                    footprints.push_back(Footprint(subFootprint, _peaks, bounds));
                }
            }
            footprint.block(bounds[0], bounds[2], subHeight, subWidth) = MatrixB::Zero(subHeight, subWidth);
        }
    }
    return footprints;
}



PYBIND11_MODULE(detect_pybind11, mod) {
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixF;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixD;

  mod.doc() = "Fast detection algorithms implemented in C++";

  mod.def("get_connected_pixels", &get_connected_pixels<MatrixF>,
          "Create a boolean mask for pixels that are connected");
  mod.def("get_connected_pixels", &get_connected_pixels<MatrixD>,
          "Create a boolean mask for pixels that are connected");

  mod.def("get_peaks", &get_peaks<MatrixF>,
          "Get a list of peaks in a footprint created by get_connected_pixels");
  mod.def("get_peaks", &get_peaks<MatrixD>,
          "Get a list of peaks in a footprint created by get_connected_pixels");

  mod.def("get_footprints", &get_footprints<MatrixF, float>,
          "Create a list of all of the footprints in an image, with their peaks"
          "image"_a, "min_separation"_a, "min_area"_a, "thresh"_a);
  mod.def("get_footprints", &get_footprints<MatrixD, double>,
          "Create a list of all of the footprints in an image, with their peaks"
          "image"_a, "min_separation"_a, "min_area"_a, "thresh"_a);

  py::class_<Footprint>(mod, "Footprint")
        .def(py::init<MatrixB, std::vector<std::vector<int>>, Bounds>(),
             "footprint"_a, "peaks"_a, "bounds"_a)
        .def_property_readonly("footprint", &Footprint::getFootprint)
        .def_property_readonly("peaks", &Footprint::getPeaks)
        .def_property_readonly("bounds", &Footprint::getBounds);
}
