#ifndef SAM3CPP__ARTIFACT_RESOLVER_H_
#define SAM3CPP__ARTIFACT_RESOLVER_H_

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ArtifactResolver {

struct ImageRuntimeSelection {
    std::string encoderPath;
    std::string decoderPath;
    std::string precision;
    std::string mode;
};

struct VideoRuntimeSelection {
    std::string encoderPath;
    std::string decoderPath;
    std::string memoryAttentionPath;
    std::string memoryEncoderPath;
    std::string constantsPath;
    std::string precision;
    std::string graphProfile;
    std::string mode;
};

inline std::string lowerCopy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline std::string preferredRuntimeProfile()
{
    const char* value = std::getenv("SAM3_ORT_RUNTIME_PROFILE");
    return value ? lowerCopy(value) : std::string();
}

inline bool isLowCostCpuProfile()
{
    const std::string profile = preferredRuntimeProfile();
    return profile == "cpu_lowcost"
        || profile == "lowcost_cpu"
        || profile == "cpu-lowcost"
        || profile == "low-cost-cpu";
}

inline int preferredRuntimeThreads(int fallback, const std::string& device)
{
    const char* explicitThreads = std::getenv("SAM3_ORT_CPU_THREADS");
    if (explicitThreads && *explicitThreads) {
        try {
            return std::max(1, std::stoi(explicitThreads));
        } catch (...) {
        }
    }

    const char* intraOpThreads = std::getenv("SAM3_ORT_INTRA_OP_THREADS");
    if (intraOpThreads && *intraOpThreads) {
        try {
            return std::max(1, std::stoi(intraOpThreads));
        } catch (...) {
        }
    }

    if (isLowCostCpuProfile() || device == "cpu") {
        return std::max(1, std::min(fallback, 4));
    }
    return std::max(1, fallback);
}

inline std::filesystem::path executablePath()
{
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    while (length >= buffer.size()) {
        buffer.resize(buffer.size() * 2);
        length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    }
    buffer.resize(length);
    return std::filesystem::path(buffer);
#else
    return std::filesystem::current_path();
#endif
}

inline bool pathExists(const std::filesystem::path& path)
{
    try {
        return std::filesystem::exists(path);
    } catch (...) {
        return false;
    }
}

inline std::filesystem::path normalizePath(const std::filesystem::path& path)
{
    try {
        if (std::filesystem::exists(path)) {
            return std::filesystem::absolute(path).lexically_normal();
        }
    } catch (...) {
    }
    return path.lexically_normal();
}

inline std::filesystem::path candidatePath(const std::string& path)
{
    if (path.empty()) {
        return std::filesystem::path();
    }
    std::filesystem::path candidate(path);
    if (candidate.is_absolute()) {
        return candidate;
    }
    return std::filesystem::current_path() / candidate;
}

inline std::vector<std::filesystem::path> repoSearchSeeds()
{
    std::vector<std::filesystem::path> seeds;
    seeds.push_back(std::filesystem::current_path());

    const std::filesystem::path exePath = executablePath();
    if (!exePath.empty()) {
        seeds.push_back(exePath.parent_path());
    }

    seeds.push_back(std::filesystem::path(__FILE__).parent_path().parent_path().parent_path());
    return seeds;
}

inline std::filesystem::path findRepoRoot()
{
    for (const auto& seed : repoSearchSeeds()) {
        std::filesystem::path current = normalizePath(seed);
        while (!current.empty()) {
            if (pathExists(current / "checkpoints" / "sam3")) {
                return current;
            }

            const auto parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
    }
    return std::filesystem::path();
}

inline std::filesystem::path defaultImageOnnxDir()
{
    const auto repoRoot = findRepoRoot();
    return repoRoot.empty() ? std::filesystem::path() : repoRoot / "checkpoints" / "sam3" / "onnx";
}

inline std::filesystem::path defaultVideoOnnxDir()
{
    const auto repoRoot = findRepoRoot();
    return repoRoot.empty() ? std::filesystem::path() : repoRoot / "checkpoints" / "sam3" / "video_onnx";
}

inline bool hasExternalDataSidecar(const std::filesystem::path& path)
{
    return pathExists(path) && pathExists(path.string() + "_data");
}

inline std::vector<std::string> preferredImagePrecisions(const std::string& device)
{
    const std::string requested = lowerCopy(std::getenv("SAM3_ONNX_VARIANT") ? std::getenv("SAM3_ONNX_VARIANT") : "auto");
    if (requested == "fp16") {
        return {"fp16", "fp32"};
    }
    if (requested == "fp32") {
        return {"fp32", "fp16"};
    }
    return device == "cpu"
        ? std::vector<std::string>{"fp32", "fp16"}
        : std::vector<std::string>{"fp16", "fp32"};
}

inline std::vector<std::string> preferredTrackerPrecisions(const std::string& device)
{
    const std::string requested = lowerCopy(
        std::getenv("SAM3_ORT_TRACKER_PRECISION")
            ? std::getenv("SAM3_ORT_TRACKER_PRECISION")
            : "auto");
    if (requested == "fp16") {
        return {"fp16", "fp32"};
    }
    if (requested == "fp32") {
        return {"fp32", "fp16"};
    }
    return device == "cpu"
        ? std::vector<std::string>{"fp32", "fp16"}
        : std::vector<std::string>{"fp16", "fp32"};
}

inline ImageRuntimeSelection resolveImageRuntimePaths(const std::string& encoderPath,
                                                      const std::string& decoderPath,
                                                      const std::string& device)
{
    const auto onnxDir = defaultImageOnnxDir();

    ImageRuntimeSelection fallback;
    fallback.encoderPath = normalizePath(candidatePath(encoderPath)).string();
    fallback.decoderPath = normalizePath(candidatePath(decoderPath)).string();
    fallback.precision = "manual";
    fallback.mode = "manual";

    if (encoderPath.empty() || decoderPath.empty()) {
        for (const auto& precision : preferredImagePrecisions(device)) {
            const std::filesystem::path encoderCandidate =
                onnxDir / (precision == "fp16" ? "vision_encoder_fp16.onnx" : "vision_encoder.onnx");
            const std::filesystem::path decoderCandidate =
                onnxDir / (precision == "fp16"
                    ? "prompt_encoder_mask_decoder_fp16.onnx"
                    : "prompt_encoder_mask_decoder.onnx");

            if (hasExternalDataSidecar(encoderCandidate) && hasExternalDataSidecar(decoderCandidate)) {
                ImageRuntimeSelection selection;
                selection.encoderPath = normalizePath(encoderCandidate).string();
                selection.decoderPath = normalizePath(decoderCandidate).string();
                selection.precision = precision;
                selection.mode = "resolved";
                if (!encoderPath.empty()) {
                    selection.encoderPath = normalizePath(candidatePath(encoderPath)).string();
                    selection.mode = "manual-encoder";
                }
                if (!decoderPath.empty()) {
                    selection.decoderPath = normalizePath(candidatePath(decoderPath)).string();
                    selection.mode = selection.mode == "resolved" ? "manual-decoder" : selection.mode + "+decoder";
                }
                return selection;
            }
        }
    }

    return fallback;
}

inline std::string graphProfileForAnnotationCount(size_t annotationCount)
{
    return annotationCount > 1 ? "multi" : "single";
}

inline VideoRuntimeSelection resolveVideoRuntimePaths(const std::string& encoderPath,
                                                      const std::string& decoderPath,
                                                      const std::string& memoryAttentionPath,
                                                      const std::string& memoryEncoderPath,
                                                      const std::string& constantsPath,
                                                      const std::string& device,
                                                      size_t annotationCount)
{
    const auto imageSelection = resolveImageRuntimePaths(encoderPath, "", device);
    const auto videoDir = defaultVideoOnnxDir();
    const std::string preferredGraphProfile = graphProfileForAnnotationCount(annotationCount);

    std::vector<std::string> graphProfiles = {preferredGraphProfile};
    if (preferredGraphProfile != "single") {
        graphProfiles.push_back("single");
    }

    for (const auto& graphProfile : graphProfiles) {
        for (const auto& precision : preferredTrackerPrecisions(device)) {
            const std::string suffix = graphProfile + (precision == "fp16" ? "_fp16" : "");

            const std::filesystem::path decoderCandidate = videoDir / ("image_decoder_" + suffix + ".onnx");
            const std::filesystem::path memoryAttentionCandidate = videoDir / ("memory_attention_" + suffix + ".onnx");
            const std::filesystem::path memoryEncoderCandidate = videoDir / ("memory_encoder_" + suffix + ".onnx");
            const std::filesystem::path constantsCandidate = videoDir / ("video_constants_" + suffix + ".npz");

            if (!pathExists(decoderCandidate)
                || !pathExists(memoryAttentionCandidate)
                || !pathExists(memoryEncoderCandidate)
                || !pathExists(constantsCandidate)) {
                continue;
            }

            VideoRuntimeSelection selection;
            selection.encoderPath = imageSelection.encoderPath;
            selection.decoderPath = normalizePath(decoderCandidate).string();
            selection.memoryAttentionPath = normalizePath(memoryAttentionCandidate).string();
            selection.memoryEncoderPath = normalizePath(memoryEncoderCandidate).string();
            selection.constantsPath = normalizePath(constantsCandidate).string();
            selection.precision = precision;
            selection.graphProfile = graphProfile;
            selection.mode = "resolved";

            if (!encoderPath.empty()) {
                selection.encoderPath = normalizePath(candidatePath(encoderPath)).string();
                selection.mode = "manual-encoder";
            }
            if (!decoderPath.empty()) {
                selection.decoderPath = normalizePath(candidatePath(decoderPath)).string();
                selection.mode = "manual-decoder";
            }
            if (!memoryAttentionPath.empty()) {
                selection.memoryAttentionPath = normalizePath(candidatePath(memoryAttentionPath)).string();
                selection.mode = "manual-memory-attention";
            }
            if (!memoryEncoderPath.empty()) {
                selection.memoryEncoderPath = normalizePath(candidatePath(memoryEncoderPath)).string();
                selection.mode = "manual-memory-encoder";
            }
            if (!constantsPath.empty()) {
                selection.constantsPath = normalizePath(candidatePath(constantsPath)).string();
                selection.mode = "manual-constants";
            }
            return selection;
        }
    }

    VideoRuntimeSelection fallback;
    fallback.encoderPath = imageSelection.encoderPath;
    fallback.decoderPath = normalizePath(candidatePath(decoderPath)).string();
    fallback.memoryAttentionPath = normalizePath(candidatePath(memoryAttentionPath)).string();
    fallback.memoryEncoderPath = normalizePath(candidatePath(memoryEncoderPath)).string();
    fallback.constantsPath = normalizePath(candidatePath(constantsPath)).string();
    fallback.precision = "manual";
    fallback.graphProfile = preferredGraphProfile;
    fallback.mode = "manual";
    return fallback;
}

} // namespace ArtifactResolver

#endif // SAM3CPP__ARTIFACT_RESOLVER_H_
