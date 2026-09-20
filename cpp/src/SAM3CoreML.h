#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace smseg_sam3 {

// Provider policy only. Model selection and segmentation remain in SAM3.
// Kept identical in sam3-onnx-cpp; no Apple SDK dependency in public headers.
struct SAM3CoreMLConfig {
    std::string computeUnits = "ALL";
    std::string specialization = "Default";
    bool requireStaticShapes = false;
    bool profileComputePlan = false;
    std::string cacheDirectory;
    std::string cacheKey;

    static std::string environment(const char* name, const char* fallback = "")
    {
        const char* value = std::getenv(name);
        return value && *value ? value : fallback;
    }

    static bool booleanEnvironment(const char* name)
    {
        const auto value = environment(name, "0");
        if (value != "0" && value != "1")
            throw std::invalid_argument(std::string(name) + " must be 0 or 1");
        return value == "1";
    }

    static SAM3CoreMLConfig fromEnvironment()
    {
        SAM3CoreMLConfig config;
        config.computeUnits = environment("SAM3_ORT_COREML_COMPUTE_UNITS", "ALL");
        config.specialization = environment("SAM3_ORT_COREML_SPECIALIZATION", "Default");
        config.requireStaticShapes = booleanEnvironment("SAM3_ORT_COREML_STATIC_SHAPES");
        config.profileComputePlan = booleanEnvironment("SAM3_ORT_COREML_PROFILE");
        config.cacheDirectory = environment("SAM3_ORT_COREML_CACHE_DIR");
        config.cacheKey = environment("SAM3_ORT_COREML_CACHE_KEY");
        config.validate();
        return config;
    }

    void validate() const
    {
        if (computeUnits != "ALL" && computeUnits != "CPUAndGPU"
            && computeUnits != "CPUAndNeuralEngine" && computeUnits != "CPUOnly")
            throw std::invalid_argument("Invalid SAM3 Core ML compute units");
        if (specialization != "Default" && specialization != "FastPrediction")
            throw std::invalid_argument("Invalid SAM3 Core ML specialization strategy");
        if (cacheDirectory.empty() != cacheKey.empty())
            throw std::invalid_argument("Core ML caching requires both a directory and a verified content key");
        if (!cacheDirectory.empty()) {
            if (!std::filesystem::path(cacheDirectory).is_absolute())
                throw std::invalid_argument("Core ML cache directory must be absolute");
            if (cacheKey.size() != 64 || !std::all_of(cacheKey.begin(), cacheKey.end(), [](char c) {
                    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                }))
                throw std::invalid_argument("Core ML cache key must be a lowercase SHA256 digest");
        }
    }

    std::unordered_map<std::string, std::string> providerOptions() const
    {
        validate();
        std::unordered_map<std::string, std::string> options{
            {"ModelFormat", "MLProgram"}, {"MLComputeUnits", computeUnits},
            {"RequireStaticInputShapes", requireStaticShapes ? "1" : "0"},
            {"EnableOnSubgraphs", "0"}, {"SpecializationStrategy", specialization},
            {"ProfileComputePlan", profileComputePlan ? "1" : "0"},
            {"AllowLowPrecisionAccumulationOnGPU", "0"},
        };
        // ORT's own cache hash can depend only on the model path. The caller
        // must verify graph + external weights + runtime + configuration and
        // supply their content digest. Never cache on a mutable path alone.
        if (!cacheDirectory.empty())
            options["ModelCacheDirectory"] = (std::filesystem::path(cacheDirectory) / cacheKey).string();
        return options;
    }
};

} // namespace smseg_sam3
