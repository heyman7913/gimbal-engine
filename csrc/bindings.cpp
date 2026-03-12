#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("correlation_forward", &correlation_forward);
  m.def("correlation_backward", &correlation_backward);
  m.def("scharr_gradient", &scharr_gradient);
  m.def("shi_tomasi_response", &shi_tomasi_response);
  m.def("gaussian_downsample", &gaussian_downsample);
  m.def("warp_bilinear", &warp_bilinear);
}
