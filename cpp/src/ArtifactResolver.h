#ifndef ARTIFACTRESOLVER_H
#define ARTIFACTRESOLVER_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace ArtifactResolver {

struct ImageDecoderSelection {
    std::string path;
    std::string mode;
};

struct ImageRuntimeSelection {
    std::string encoderPath;
    std::string decoderPath;
    std::string precision;
    std::string graphProfile;
    std::string mode;
};

struct VideoRuntimeSelection {
    std::string decoderInitPath;
    std::string decoderPropagatePath;
    std::string memoryAttentionPath;
    std::string memoryEncoderPath;
    std::string mode;
    std::string encoderPath;
    std::string decoderPath;
    std::string constantsPath;
    std::string precision;
    std::string graphProfile;
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

inline std::filesystem::path candidatePath(const std::string &path)
{
    std::filesystem::path candidate(path);
    if (candidate.has_parent_path()) {
        return candidate;
    }
    return std::filesystem::current_path() / candidate;
}

inline std::string normalizePath(const std::filesystem::path &path)
{
    try {
        if (std::filesystem::exists(path)) {
            return std::filesystem::absolute(path).lexically_normal().string();
        }
    } catch (...) {
    }
    return path.lexically_normal().string();
}

inline const char *envValue(const char *primary, const char *fallback = nullptr)
{
    const char *value = std::getenv(primary);
    if (value && *value) {
        return value;
    }
    return fallback ? std::getenv(fallback) : nullptr;
}

inline std::string preferredRuntimeProfile();
inline bool isLowCostCpuProfile();

inline std::string preferredRuntimeProfile()
{
    const char *value = envValue("SAM3_ORT_RUNTIME_PROFILE", "SAM2_ORT_RUNTIME_PROFILE");
    if (!value) {
        return "";
    }
    return lowerCopy(value);
}

inline bool isLowCostCpuProfile()
{
    const std::string profile = preferredRuntimeProfile();
    return profile == "cpu_lowcost"
        || profile == "lowcost_cpu"
        || profile == "cpu-lowcost"
        || profile == "low-cost-cpu";
}

inline int preferredRuntimeThreads(int fallback, const std::string &device)
{
    const char *value = envValue("SAM3_ORT_CPU_THREADS");
    if (!value) {
        value = envValue("SAM3_ORT_INTRA_OP_THREADS");
    }
    if (!value) {
        value = envValue("SAM2_ORT_CPU_THREADS", "SAM2_ORT_INTRA_OP_THREADS");
    }
    if (value && *value) {
        try {
            const int requested = std::stoi(value);
            // Zero is an intentional value: it leaves ORT's intra-op thread
            // count at its platform default. Negative values are invalid and
            // fall through to the normal profile/device policy.
            if (requested >= 0) {
                return requested;
            }
        } catch (...) {
        }
    }

    if (isLowCostCpuProfile() || lowerCopy(device).rfind("cpu", 0) == 0) {
        return std::max(1, std::min(fallback, 4));
    }
    return std::max(1, fallback);
}

inline std::string normalizedVariant(const char *value,
                                     const std::string &fallback = "auto")
{
    if (!value || !*value) {
        return fallback;
    }
    const std::string lowered = lowerCopy(value);
    if (lowered == "int8" || lowered == "fp32" || lowered == "fp16"
        || lowered == "auto") {
        return lowered;
    }
    return fallback;
}

inline std::string preferredEncoderVariant()
{
    const char *value = envValue("SAM3_ORT_ENCODER_VARIANT");
    if (!value) {
        value = envValue("SAM3_ONNX_VARIANT");
    }
    if (!value) {
        value = envValue("SAM2_ORT_ENCODER_VARIANT", "SAM2_ONNX_VARIANT");
    }
    return normalizedVariant(value);
}

inline std::string preferredDecoderVariant()
{
    const char *value = envValue("SAM3_ORT_DECODER_VARIANT");
    if (!value) {
        value = envValue("SAM3_ONNX_VARIANT");
    }
    if (!value) {
        value = envValue("SAM2_ORT_DECODER_VARIANT", "SAM2_ONNX_VARIANT");
    }
    return normalizedVariant(value);
}

inline std::string preferredVideoModuleVariant()
{
    const char *value = envValue("SAM3_ORT_TRACKER_PRECISION");
    if (!value) {
        value = envValue("SAM3_ORT_VIDEO_MODULE_VARIANT");
    }
    if (!value) {
        value = envValue("SAM2_ORT_TRACKER_PRECISION", "SAM2_ORT_VIDEO_MODULE_VARIANT");
    }
    return normalizedVariant(value, "auto");
}

inline bool useExperimentalVideoInitDecoder()
{
    const char *value = envValue(
        "SAM3_ORT_EXPERIMENTAL_VIDEO_INIT_DECODER",
        "SAM2_ORT_EXPERIMENTAL_VIDEO_INIT_DECODER");
    if (!value) {
        return false;
    }
    const std::string lowered = lowerCopy(value);
    return lowered == "1" || lowered == "true" || lowered == "yes";
}

inline bool pathExists(const std::filesystem::path &path)
{
    try {
        return std::filesystem::exists(path);
    } catch (...) {
        return false;
    }
}

inline bool isBasename(const std::string &path, const std::string &name)
{
    return lowerCopy(candidatePath(path).filename().string()) == lowerCopy(name);
}

inline std::filesystem::path variantArtifactPath(const std::filesystem::path &base,
                                                 const std::string &variant)
{
    if (variant == "fp16") {
        return base.parent_path()
            / (base.stem().string() + "_fp16" + base.extension().string());
    }
    if (variant == "int8") {
        return base.parent_path()
            / (base.stem().string() + ".int8" + base.extension().string());
    }
    return base;
}

inline std::string selectRuntimeArtifactVariant(const std::string &path,
                                                const std::string &device,
                                                const std::string &variant)
{
    const std::filesystem::path base = candidatePath(path);
    if (lowerCopy(base.extension().string()) != ".onnx") {
        return path;
    }

    const std::filesystem::path fp16 = variantArtifactPath(base, "fp16");
    const std::filesystem::path int8 = variantArtifactPath(base, "int8");
    const auto useIfPresent = [&](const std::filesystem::path &candidate) -> std::string {
        return pathExists(candidate) ? normalizePath(candidate) : std::string();
    };

    if (variant == "fp32") {
        return path;
    }
    if (variant == "fp16") {
        const std::string selected = useIfPresent(fp16);
        return selected.empty() ? path : selected;
    }
    if (variant == "int8") {
        const std::string selected = useIfPresent(int8);
        return selected.empty() ? path : selected;
    }

    const bool cpu = lowerCopy(device).rfind("cpu", 0) == 0;
    if (cpu) {
        const std::string selected = useIfPresent(int8);
        if (!selected.empty()) {
            return selected;
        }
        if (pathExists(base)) {
            return normalizePath(base);
        }
        const std::string fallback = useIfPresent(fp16);
        return fallback.empty() ? path : fallback;
    }

    const std::string selected = useIfPresent(fp16);
    if (!selected.empty()) {
        return selected;
    }
    if (pathExists(base)) {
        return normalizePath(base);
    }
    return path;
}

inline std::string preferQuantizedEncoderPath(const std::string &encoderPath,
                                              const std::string &device)
{
    if (!isBasename(encoderPath, "image_encoder.onnx")
        && !isBasename(encoderPath, "vision_encoder.onnx")) {
        return encoderPath;
    }
    return selectRuntimeArtifactVariant(
        encoderPath,
        device,
        preferredEncoderVariant());
}

inline std::string preferQuantizedRuntimeArtifactPath(const std::string &path,
                                                      const std::string &device,
                                                      const std::string &variant)
{
    return selectRuntimeArtifactVariant(path, device, variant);
}

inline std::string precisionLabelForPath(const std::string &path)
{
    const std::string lowered = lowerCopy(path);
    if (lowered.find("int8") != std::string::npos) {
        return "int8";
    }
    if (lowered.find("fp16") != std::string::npos) {
        return "fp16";
    }
    return "fp32";
}

inline ImageDecoderSelection resolveImageDecoderPath(const std::string &decoderPath,
                                                     const std::string &promptMode,
                                                     bool experimentalImagePointDecoder = false)
{
    if (!isBasename(decoderPath, "image_decoder.onnx")) {
        return {decoderPath, "manual"};
    }

    const std::filesystem::path current = candidatePath(decoderPath);
    const std::filesystem::path directory = current.parent_path();

    if (promptMode == "bounding_box") {
        const std::filesystem::path specialized = directory / "image_decoder_box.onnx";
        if (pathExists(specialized)) {
            return {normalizePath(specialized), "specialized"};
        }
        return {decoderPath, "legacy"};
    }

    if (promptMode == "seed_points") {
        const std::filesystem::path specialized = directory / "image_decoder_points.onnx";
        if (experimentalImagePointDecoder && pathExists(specialized)) {
            return {normalizePath(specialized), "specialized"};
        }
        return {decoderPath, experimentalImagePointDecoder ? "legacy-missing-specialized" : "legacy-safe-seed-points"};
    }

    return {decoderPath, "legacy"};
}

inline ImageRuntimeSelection resolveImageRuntimePaths(const std::string &encoderPath,
                                                      const std::string &decoderPath,
                                                      const std::string &device = "cpu")
{
    const std::string selectedEncoder = preferQuantizedEncoderPath(encoderPath, device);
    const std::string selectedDecoderArtifact =
        (isBasename(decoderPath, "prompt_encoder_mask_decoder.onnx")
         || isBasename(decoderPath, "image_decoder.onnx"))
            ? selectRuntimeArtifactVariant(
                  decoderPath,
                  device,
                  preferredDecoderVariant())
            : decoderPath;
    const ImageDecoderSelection selectedDecoder =
        resolveImageDecoderPath(selectedDecoderArtifact, "seed_points", false);
    return {
        selectedEncoder,
        selectedDecoder.path,
        precisionLabelForPath(selectedEncoder),
        selectedDecoder.mode,
        selectedDecoder.mode,
    };
}

inline VideoRuntimeSelection resolveVideoRuntimePaths(const std::string &decoderPath,
                                                      const std::string &memoryAttentionPath,
                                                      const std::string &memoryEncoderPath,
                                                      bool experimentalOneFrameAttention = false,
                                                      const std::string &device = "cpu")
{
    const std::filesystem::path decoderCandidate = candidatePath(decoderPath);
    const std::filesystem::path decoderDirectory = decoderCandidate.parent_path();
    const std::filesystem::path manualDecoderInit = decoderDirectory / "video_decoder_init.onnx";
    const std::filesystem::path manualDecoderPropagate = decoderDirectory / "video_decoder_propagate.onnx";
    const std::string videoModuleVariant = preferredVideoModuleVariant();

    const auto applyVideoModuleVariants = [&](VideoRuntimeSelection selection) -> VideoRuntimeSelection {
        selection.decoderPropagatePath = preferQuantizedRuntimeArtifactPath(
            selection.decoderPropagatePath,
            device,
            videoModuleVariant);
        selection.memoryAttentionPath = preferQuantizedRuntimeArtifactPath(
            selection.memoryAttentionPath,
            device,
            videoModuleVariant);
        selection.memoryEncoderPath = preferQuantizedRuntimeArtifactPath(
            selection.memoryEncoderPath,
            device,
            videoModuleVariant);
        return selection;
    };

    if ((isBasename(decoderPath, "video_decoder_init.onnx")
         || isBasename(decoderPath, "video_decoder_propagate.onnx"))
        && (isBasename(memoryAttentionPath, "memory_attention_objptr.onnx")
            || isBasename(memoryAttentionPath, "memory_attention_no_objptr.onnx")
            || isBasename(memoryAttentionPath, "memory_attention_no_objptr_1frame.onnx"))
        && (isBasename(memoryEncoderPath, "memory_encoder.onnx")
            || isBasename(memoryEncoderPath, "memory_encoder_lite.onnx"))) {
        const std::string mode = (isBasename(memoryEncoderPath, "memory_encoder.onnx")
                                      ? "manual-specialized-temporal"
                                      : "manual-specialized-lite");
        return applyVideoModuleVariants({
            normalizePath(isBasename(decoderPath, "video_decoder_init.onnx") ? decoderCandidate : manualDecoderInit),
            normalizePath(isBasename(decoderPath, "video_decoder_propagate.onnx") ? decoderCandidate : manualDecoderPropagate),
            normalizePath(candidatePath(memoryAttentionPath)),
            normalizePath(candidatePath(memoryEncoderPath)),
            mode,
        });
    }

    if (!isBasename(decoderPath, "image_decoder.onnx")
        || !isBasename(memoryAttentionPath, "memory_attention.onnx")
        || !isBasename(memoryEncoderPath, "memory_encoder.onnx")) {
        return applyVideoModuleVariants({decoderPath, decoderPath, memoryAttentionPath, memoryEncoderPath, "manual"});
    }

    const std::filesystem::path directory = decoderDirectory;

    const std::filesystem::path decoderInit = directory / "video_decoder_init.onnx";
    const std::filesystem::path decoderProp = directory / "video_decoder_propagate.onnx";
    const std::filesystem::path attnObjPtr = directory / "memory_attention_objptr.onnx";
    const std::filesystem::path attn1Frame = directory / "memory_attention_no_objptr_1frame.onnx";
    const std::filesystem::path attnDynamic = directory / "memory_attention_no_objptr.onnx";
    const std::filesystem::path legacyMemEncoder = directory / "memory_encoder.onnx";
    const std::filesystem::path memEncoderLite = directory / "memory_encoder_lite.onnx";

    const bool specializedAvailable =
        pathExists(decoderInit)
        && pathExists(decoderProp)
        && (pathExists(attnObjPtr) || pathExists(attn1Frame) || pathExists(attnDynamic));
    const bool legacyAvailable = pathExists(candidatePath(decoderPath))
        && pathExists(candidatePath(memoryAttentionPath))
        && pathExists(candidatePath(memoryEncoderPath));
    const bool hybridAvailable =
        legacyAvailable
        && pathExists(decoderProp);
    const bool preferOptimized = device != "cpu";
    const auto hybridResult = [&]() -> VideoRuntimeSelection {
        return applyVideoModuleVariants({
            normalizePath(candidatePath(decoderPath)),
            normalizePath(decoderProp),
            normalizePath(candidatePath(memoryAttentionPath)),
            normalizePath(candidatePath(memoryEncoderPath)),
            "hybrid-propagate",
        });
    };

    const auto specializedResult = [&]() -> VideoRuntimeSelection {
        std::filesystem::path selectedAttention;
        std::string suffix;

        if (pathExists(attnObjPtr)) {
            selectedAttention = attnObjPtr;
            suffix = "objptr";
        } else if (experimentalOneFrameAttention && pathExists(attn1Frame)) {
            selectedAttention = attn1Frame;
            suffix = "1frame-attn";
        } else if (pathExists(attnDynamic)) {
            selectedAttention = attnDynamic;
            suffix = "dynamic-attn";
        } else {
            selectedAttention = attn1Frame;
            suffix = "1frame-attn-fallback";
        }

        const std::filesystem::path selectedMemEncoder = pathExists(legacyMemEncoder) ? legacyMemEncoder : memEncoderLite;
        const std::string modePrefix = pathExists(legacyMemEncoder) ? "specialized-temporal" : "specialized-lite";

        return applyVideoModuleVariants({
            normalizePath(decoderInit),
            normalizePath(decoderProp),
            normalizePath(selectedAttention),
            normalizePath(selectedMemEncoder),
            modePrefix + "-" + suffix,
        });
    };

    const auto optimizedResult = [&]() -> VideoRuntimeSelection {
        if (specializedAvailable) {
            return specializedResult();
        }
        return hybridResult();
    };

    if (preferOptimized && (hybridAvailable || specializedAvailable)) {
        return optimizedResult();
    }

    if (legacyAvailable) {
        return applyVideoModuleVariants({decoderPath, decoderPath, memoryAttentionPath, memoryEncoderPath, "legacy"});
    }

    if (hybridAvailable || specializedAvailable) {
        return optimizedResult();
    }

    return applyVideoModuleVariants({decoderPath, decoderPath, memoryAttentionPath, memoryEncoderPath, "legacy"});
}

inline VideoRuntimeSelection resolveVideoRuntimePaths(const std::string &encoderPath,
                                                      const std::string &decoderPath,
                                                      const std::string &memoryAttentionPath,
                                                      const std::string &memoryEncoderPath,
                                                      const std::string &constantsPath,
                                                      const std::string &device = "cpu",
                                                      std::size_t anchorCount = 0)
{
    const std::string selectedEncoder = preferQuantizedEncoderPath(encoderPath, device);
    VideoRuntimeSelection selection = resolveVideoRuntimePaths(
        decoderPath,
        memoryAttentionPath,
        memoryEncoderPath,
        anchorCount <= 1,
        device);
    selection.encoderPath = selectedEncoder;
    selection.decoderPath = selection.decoderPropagatePath.empty()
        ? decoderPath
        : selection.decoderPropagatePath;
    selection.constantsPath = constantsPath;
    selection.precision = precisionLabelForPath(selectedEncoder);
    selection.graphProfile = selection.mode;
    return selection;
}

} // namespace ArtifactResolver

#endif // ARTIFACTRESOLVER_H
