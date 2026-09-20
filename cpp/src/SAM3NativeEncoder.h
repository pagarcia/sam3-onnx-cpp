#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace smseg_sam3 {

// One complete, compiled Core ML encoder. Tracking continues through ONNX Runtime.
// No Apple framework types escape into portable callers. A session is not reentrant.
class SAM3NativeEncoder {
public:
    virtual ~SAM3NativeEncoder() = default;
    virtual std::vector<Ort::Value> run(const std::vector<float>& nchw) = 0;
    static std::unique_ptr<SAM3NativeEncoder> create(const std::string& compiledModel,
                                                   bool cpuOnly = false);
};

#ifndef __APPLE__
inline std::unique_ptr<SAM3NativeEncoder> SAM3NativeEncoder::create(const std::string&, bool)
{
    throw std::runtime_error("Native Core ML encoding requires macOS");
}
#endif

} // namespace smseg_sam3
