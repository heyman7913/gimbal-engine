#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("correlation_forward", &correlation_forward);
  m.def("correlation_backward", &correlation_backward);
  m.def("correlation_forward_v1", &correlation_forward_v1);
  m.def("correlation_forward_v2", &correlation_forward_v2);
  m.def("scharr_gradient", &scharr_gradient);
  m.def("shi_tomasi_response", &shi_tomasi_response);
  m.def("gaussian_downsample", &gaussian_downsample);
  m.def("warp_bilinear", &warp_bilinear);
}
