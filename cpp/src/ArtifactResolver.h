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

inline bool pathExists(const std::filesystem::path& path);
inline std::filesystem::path normalizePath(const std::filesystem::path& path);

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

inline bool isDirectMLDevice(const std::string& device)
{
    const std::string lowered = lowerCopy(device);
    return lowered.rfind("dml", 0) == 0 || lowered.rfind("directml", 0) == 0;
}

inline std::filesystem::path preferDirectMLArtifactPath(const std::filesystem::path& path,
                                                        const std::string& device)
{
    if (!isDirectMLDevice(device) || path.empty() || lowerCopy(path.extension().string()) != ".onnx") {
        return path;
    }

    const std::filesystem::path dmlPath =
        path.parent_path() / (path.stem().string() + ".dml.onnx");
    if (pathExists(dmlPath)) {
        return normalizePath(dmlPath);
    }

    return path;
}

inline int preferredRuntimeThreads(int fallback, const std::string& device)
{
    const char* explicitThreads = std::getenv("SAM3_ORT_CPU_THREADS");
    if (explicitThreads && *explicitThreads) {
        try {
            return std::max(0, std::stoi(explicitThreads));
        } catch (...) {
        }
    }

    const char* intraOpThreads = std::getenv("SAM3_ORT_INTRA_OP_THREADS");
    if (intraOpThreads && *intraOpThreads) {
        try {
            return std::max(0, std::stoi(intraOpThreads));
        } catch (...) {
        }
    }

    if (isLowCostCpuProfile()) {
        return std::max(1, std::min(fallback, 4));
    }
    if (device == "cpu") {
        return std::max(1, fallback > 1 ? fallback - 1 : fallback);
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

inline bool hasUsableImageArtifact(const std::filesystem::path& path,
                                   const std::string& variant)
{
    if (!pathExists(path)) {
        return false;
    }

    // Downloaded fp32/fp16 Hugging Face image models are split across
    // .onnx + .onnx_data. Quantized artifacts may be embedded or may carry
    // their own external-data reference, so let ONNX Runtime validate them.
    if (variant == "int8") {
        return true;
    }

    return pathExists(path.string() + "_data");
}

inline std::string preferredEncoderVariant()
{
    const char* value = std::getenv("SAM3_ORT_ENCODER_VARIANT");
    if (!value || !*value) {
        value = std::getenv("SAM3_ONNX_VARIANT");
    }
    if (!value || !*value) {
        return "auto";
    }

    const std::string requested = lowerCopy(value);
    if (requested == "int8" || requested == "fp32" || requested == "fp16" || requested == "auto") {
        return requested;
    }
    return "auto";
}

inline std::string preferredDecoderVariant()
{
    const char* value = std::getenv("SAM3_ORT_DECODER_VARIANT");
    if (!value || !*value) {
        value = std::getenv("SAM3_ONNX_VARIANT");
    }
    if (!value || !*value) {
        return "auto";
    }

    const std::string requested = lowerCopy(value);
    if (requested == "int8" || requested == "fp32" || requested == "fp16" || requested == "auto") {
        return requested;
    }
    return "auto";
}

inline std::vector<std::string> preferredImageEncoderVariants(const std::string& device)
{
    const std::string requested = preferredEncoderVariant();
    if (requested == "int8") {
        return {"int8", "fp32", "fp16"};
    }
    if (requested == "fp32") {
        return {"fp32", "int8", "fp16"};
    }
    if (requested == "fp16") {
        return {"fp16", "fp32", "int8"};
    }
    return device == "cpu"
        ? std::vector<std::string>{"int8", "fp32", "fp16"}
        : std::vector<std::string>{"fp16", "fp32", "int8"};
}

inline std::vector<std::string> preferredImageDecoderVariants(const std::string& device)
{
    const std::string requested = preferredDecoderVariant();
    if (requested == "int8") {
        return {"int8", "fp32", "fp16"};
    }
    if (requested == "fp32") {
        return {"fp32", "int8", "fp16"};
    }
    if (requested == "fp16") {
        return {"fp16", "fp32", "int8"};
    }
    return device == "cpu"
        ? std::vector<std::string>{"fp32", "int8", "fp16"}
        : std::vector<std::string>{"fp16", "fp32", "int8"};
}

inline std::vector<std::filesystem::path> imageEncoderPathsForVariant(const std::filesystem::path& onnxDir,
                                                                      const std::string& variant)
{
    if (variant == "int8") {
        return {
            onnxDir / "vision_encoder.int8.onnx",
            onnxDir / "bench_cpu" / "vision_encoder.int8.matmul_gather.onnx",
            onnxDir / "bench_cpu" / "vision_encoder.int8.matmul_gather_pre.onnx",
            onnxDir / "bench_cpu" / "vision_encoder.int8.matmul.onnx",
        };
    }
    if (variant == "fp16") {
        return {onnxDir / "vision_encoder_fp16.onnx"};
    }
    return {onnxDir / "vision_encoder.onnx"};
}

inline std::vector<std::filesystem::path> imageDecoderPathsForVariant(const std::filesystem::path& onnxDir,
                                                                      const std::string& variant)
{
    if (variant == "int8") {
        return {
            onnxDir / "prompt_encoder_mask_decoder.int8.onnx",
            onnxDir / "bench_cpu" / "prompt_encoder_mask_decoder.int8.matmul_gemm.onnx",
        };
    }
    if (variant == "fp16") {
        return {onnxDir / "prompt_encoder_mask_decoder_fp16.onnx"};
    }
    return {onnxDir / "prompt_encoder_mask_decoder.onnx"};
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
    fallback.encoderPath = encoderPath.empty() ? std::string() : normalizePath(candidatePath(encoderPath)).string();
    fallback.decoderPath = decoderPath.empty() ? std::string() : normalizePath(candidatePath(decoderPath)).string();
    fallback.precision = "manual";
    fallback.mode = "manual";

    std::filesystem::path resolvedEncoderCandidate;
    std::string resolvedEncoderVariant = "manual";
    if (encoderPath.empty()) {
        for (const auto& variant : preferredImageEncoderVariants(device)) {
            for (const auto& candidate : imageEncoderPathsForVariant(onnxDir, variant)) {
                if (hasUsableImageArtifact(candidate, variant)) {
                    resolvedEncoderCandidate = candidate;
                    resolvedEncoderVariant = variant;
                    break;
                }
            }
            if (!resolvedEncoderCandidate.empty()) {
                break;
            }
        }
    } else {
        resolvedEncoderCandidate = candidatePath(encoderPath);
    }

    std::filesystem::path resolvedDecoderCandidate;
    std::string resolvedDecoderVariant = "manual";
    if (decoderPath.empty()) {
        for (const auto& variant : preferredImageDecoderVariants(device)) {
            for (const auto& candidate : imageDecoderPathsForVariant(onnxDir, variant)) {
                if (hasUsableImageArtifact(candidate, variant)) {
                    resolvedDecoderCandidate = candidate;
                    resolvedDecoderVariant = variant;
                    break;
                }
            }
            if (!resolvedDecoderCandidate.empty()) {
                break;
            }
        }
    } else {
        resolvedDecoderCandidate = candidatePath(decoderPath);
    }

    if (!resolvedEncoderCandidate.empty() && !resolvedDecoderCandidate.empty()) {
        ImageRuntimeSelection selection;
        selection.encoderPath = preferDirectMLArtifactPath(
            normalizePath(resolvedEncoderCandidate),
            device).string();
        selection.decoderPath = preferDirectMLArtifactPath(
            normalizePath(resolvedDecoderCandidate),
            device).string();
        selection.precision = resolvedEncoderVariant == resolvedDecoderVariant
            ? resolvedEncoderVariant
            : ("enc=" + resolvedEncoderVariant + ",dec=" + resolvedDecoderVariant);
        selection.mode = (!encoderPath.empty() && !decoderPath.empty())
            ? "manual"
            : (!encoderPath.empty() ? "manual-encoder" : (!decoderPath.empty() ? "manual-decoder" : "resolved"));
        return selection;
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
    const auto imageOnnxDir = defaultImageOnnxDir();
    const auto videoDir = defaultVideoOnnxDir();
    const std::string preferredGraphProfile = graphProfileForAnnotationCount(annotationCount);

    std::filesystem::path resolvedEncoderCandidate;
    if (encoderPath.empty()) {
        for (const auto& variant : preferredImageEncoderVariants(device)) {
            for (const auto& candidate : imageEncoderPathsForVariant(imageOnnxDir, variant)) {
                if (hasUsableImageArtifact(candidate, variant)) {
                    resolvedEncoderCandidate = candidate;
                    break;
                }
            }
            if (!resolvedEncoderCandidate.empty()) {
                break;
            }
        }
    } else {
        resolvedEncoderCandidate = candidatePath(encoderPath);
    }

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
                || !pathExists(constantsCandidate)
                || resolvedEncoderCandidate.empty()) {
                continue;
            }

            VideoRuntimeSelection selection;
            selection.encoderPath = preferDirectMLArtifactPath(
                normalizePath(resolvedEncoderCandidate),
                device).string();
            selection.decoderPath = preferDirectMLArtifactPath(
                normalizePath(decoderCandidate),
                device).string();
            selection.memoryAttentionPath = preferDirectMLArtifactPath(
                normalizePath(memoryAttentionCandidate),
                device).string();
            selection.memoryEncoderPath = preferDirectMLArtifactPath(
                normalizePath(memoryEncoderCandidate),
                device).string();
            selection.constantsPath = normalizePath(constantsCandidate).string();
            selection.precision = precision;
            selection.graphProfile = graphProfile;
            selection.mode = "resolved";

            if (!encoderPath.empty()) {
                selection.encoderPath = preferDirectMLArtifactPath(
                    normalizePath(candidatePath(encoderPath)),
                    device).string();
                selection.mode = "manual-encoder";
            }
            if (!decoderPath.empty()) {
                selection.decoderPath = preferDirectMLArtifactPath(
                    normalizePath(candidatePath(decoderPath)),
                    device).string();
                selection.mode = "manual-decoder";
            }
            if (!memoryAttentionPath.empty()) {
                selection.memoryAttentionPath = preferDirectMLArtifactPath(
                    normalizePath(candidatePath(memoryAttentionPath)),
                    device).string();
                selection.mode = "manual-memory-attention";
            }
            if (!memoryEncoderPath.empty()) {
                selection.memoryEncoderPath = preferDirectMLArtifactPath(
                    normalizePath(candidatePath(memoryEncoderPath)),
                    device).string();
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
    fallback.encoderPath = resolvedEncoderCandidate.empty()
        ? std::string()
        : preferDirectMLArtifactPath(
            normalizePath(resolvedEncoderCandidate),
            device).string();
    fallback.decoderPath = preferDirectMLArtifactPath(
        normalizePath(candidatePath(decoderPath)),
        device).string();
    fallback.memoryAttentionPath = preferDirectMLArtifactPath(
        normalizePath(candidatePath(memoryAttentionPath)),
        device).string();
    fallback.memoryEncoderPath = preferDirectMLArtifactPath(
        normalizePath(candidatePath(memoryEncoderPath)),
        device).string();
    fallback.constantsPath = normalizePath(candidatePath(constantsPath)).string();
    fallback.precision = "manual";
    fallback.graphProfile = preferredGraphProfile;
    fallback.mode = "manual";
    return fallback;
}

} // namespace ArtifactResolver

#endif // SAM3CPP__ARTIFACT_RESOLVER_H_
