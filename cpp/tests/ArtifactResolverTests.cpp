#include "ArtifactResolver.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void setEnv(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct EnvGuard {
    explicit EnvGuard(const char* envName)
        : name(envName)
    {
        const char* current = std::getenv(name.c_str());
        if (current) {
            hadValue = true;
            value = current;
        }
    }

    ~EnvGuard()
    {
        setEnv(name.c_str(), hadValue ? value.c_str() : nullptr);
    }

    std::string name;
    std::string value;
    bool hadValue = false;
};

bool expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
    }
    return condition;
}

void touch(const std::filesystem::path& path)
{
    std::ofstream(path, std::ios::binary).put('\0');
}

std::string fileName(const std::string& path)
{
    return ArtifactResolver::lowerCopy(
        std::filesystem::path(path).filename().string());
}

} // namespace

int main()
{
    EnvGuard encoderVariant("SAM3_ORT_ENCODER_VARIANT");
    EnvGuard decoderVariant("SAM3_ORT_DECODER_VARIANT");
    EnvGuard globalVariant("SAM3_ONNX_VARIANT");
    EnvGuard trackerVariant("SAM3_ORT_TRACKER_PRECISION");
    EnvGuard legacyTrackerVariant("SAM3_ORT_VIDEO_MODULE_VARIANT");
    EnvGuard cpuThreads("SAM3_ORT_CPU_THREADS");
    EnvGuard intraThreads("SAM3_ORT_INTRA_OP_THREADS");

    setEnv("SAM3_ORT_ENCODER_VARIANT", nullptr);
    setEnv("SAM3_ORT_DECODER_VARIANT", nullptr);
    setEnv("SAM3_ONNX_VARIANT", nullptr);
    setEnv("SAM3_ORT_TRACKER_PRECISION", nullptr);
    setEnv("SAM3_ORT_VIDEO_MODULE_VARIANT", nullptr);
    setEnv("SAM3_ORT_CPU_THREADS", nullptr);
    setEnv("SAM3_ORT_INTRA_OP_THREADS", nullptr);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "sam3_artifact_resolver_tests";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root);

    const auto encoder = root / "vision_encoder.onnx";
    const auto encoderFp16 = root / "vision_encoder_fp16.onnx";
    const auto encoderInt8 = root / "vision_encoder.int8.onnx";
    const auto decoder = root / "prompt_encoder_mask_decoder.onnx";
    const auto decoderFp16 = root / "prompt_encoder_mask_decoder_fp16.onnx";
    const auto attention = root / "memory_attention_objptr.onnx";
    const auto attentionFp16 = root / "memory_attention_objptr_fp16.onnx";
    touch(encoder);
    touch(encoderFp16);
    touch(encoderInt8);
    touch(decoder);
    touch(decoderFp16);
    touch(attention);
    touch(attentionFp16);

    bool ok = true;
    ok &= expect(
        fileName(ArtifactResolver::preferQuantizedEncoderPath(encoder.string(), "cpu"))
            == "vision_encoder.int8.onnx",
        "CPU auto mode should prefer the current vision_encoder.int8.onnx name");
    ok &= expect(
        fileName(ArtifactResolver::preferQuantizedEncoderPath(encoder.string(), "cuda:0"))
            == "vision_encoder_fp16.onnx",
        "accelerator auto mode should prefer vision_encoder_fp16.onnx");

    setEnv("SAM3_ORT_ENCODER_VARIANT", "fp32");
    ok &= expect(
        fileName(ArtifactResolver::preferQuantizedEncoderPath(encoder.string(), "cuda:0"))
            == "vision_encoder.onnx",
        "explicit encoder fp32 should keep the base artifact");
    setEnv("SAM3_ORT_ENCODER_VARIANT", nullptr);

    setEnv("SAM3_ONNX_VARIANT", "fp16");
    const auto imageSelection = ArtifactResolver::resolveImageRuntimePaths(
        encoder.string(), decoder.string(), "cuda:0");
    ok &= expect(fileName(imageSelection.encoderPath) == "vision_encoder_fp16.onnx",
                 "global fp16 should select the FP16 encoder");
    ok &= expect(fileName(imageSelection.decoderPath)
                     == "prompt_encoder_mask_decoder_fp16.onnx",
                 "global fp16 should select the FP16 prompt decoder");
    setEnv("SAM3_ONNX_VARIANT", nullptr);

    setEnv("SAM3_ORT_TRACKER_PRECISION", "fp16");
    ok &= expect(
        fileName(ArtifactResolver::preferQuantizedRuntimeArtifactPath(
            attention.string(), "cuda:0", ArtifactResolver::preferredVideoModuleVariant()))
            == "memory_attention_objptr_fp16.onnx",
        "tracker fp16 should use the exported _fp16 naming convention");
    setEnv("SAM3_ORT_TRACKER_PRECISION", nullptr);

    setEnv("SAM3_ORT_CPU_THREADS", "0");
    ok &= expect(ArtifactResolver::preferredRuntimeThreads(12, "cpu") == 0,
                 "SAM3_ORT_CPU_THREADS=0 must preserve ORT's default");
    setEnv("SAM3_ORT_CPU_THREADS", nullptr);
    setEnv("SAM3_ORT_INTRA_OP_THREADS", "0");
    ok &= expect(ArtifactResolver::preferredRuntimeThreads(12, "cpu") == 0,
                 "SAM3_ORT_INTRA_OP_THREADS=0 must be a supported fallback");
    setEnv("SAM3_ORT_INTRA_OP_THREADS", nullptr);
    ok &= expect(ArtifactResolver::preferredRuntimeThreads(12, "cpu") == 4,
                 "default CPU policy should remain capped at four threads");

    std::filesystem::remove_all(root, error);
    if (ok) {
        std::cout << "[PASS] ArtifactResolver tests\n";
        return 0;
    }
    return 1;
}
