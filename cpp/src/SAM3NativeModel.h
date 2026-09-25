#pragma once
#include <filesystem>
#include <cstdlib>
#include <string>

namespace smseg_sam3 {
inline std::filesystem::path nativeModelPath(const std::string& value) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(value.begin(), value.end()));
#else
    return std::filesystem::u8path(value);
#endif
}
inline std::string nativeModelPathString(const std::filesystem::path& value) {
    const auto utf8=value.u8string();
    return std::string(utf8.begin(),utf8.end());
}
// Packaging places the verified optional export beside the ONNX bundle.
// The native loader still validates the compiled model's tensor contract.
inline std::string findNativeEncoderModel(const std::string& onnxEncoder,
                             const char* configured = std::getenv("SAM3_NATIVE_ENCODER_MODEL")) {
    const auto compiled=[](const std::filesystem::path& path) {
        std::error_code error;
        return path.extension()==".mlmodelc" && std::filesystem::is_directory(path,error);
    };
    if (configured && *configured) {
        const auto path=nativeModelPath(configured);
        return compiled(path) ? nativeModelPathString(path) : std::string{};
    }
    if (onnxEncoder.empty()) return {};
    const auto encoder=nativeModelPath(onnxEncoder);
    if (compiled(encoder)) return nativeModelPathString(encoder);
    // Root bundle and the optional fp16/fp32 subdirectory layouts.
    const auto folder=encoder.parent_path();
    for (const auto& root : {folder,folder.parent_path()}) {
        const auto path=root / "coreml" / "encoder.mlmodelc";
        if (compiled(path)) return nativeModelPathString(path);
    }
    return {};
}
}
