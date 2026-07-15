#ifndef SAM3CPP__SAM3_H_
#define SAM3CPP__SAM3_H_

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "Image.h"

namespace smseg_sam3 {

inline size_t computeElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (const auto dim : shape) {
        count *= static_cast<size_t>(std::max<int64_t>(dim, 0));
    }
    return count;
}

template <typename T>
inline Ort::Value createTensor(const Ort::MemoryInfo& memoryInfo,
                               const std::vector<T>& data,
                               const std::vector<int64_t>& shape)
{
    return Ort::Value::CreateTensor<T>(
        memoryInfo,
        const_cast<T*>(data.data()),
        data.size(),
        shape.data(),
        shape.size());
}

template <typename T>
inline Ort::Value createTensorView(const Ort::MemoryInfo& memoryInfo,
                                   T* data,
                                   const std::vector<int64_t>& shape)
{
    return Ort::Value::CreateTensor<T>(
        memoryInfo,
        data,
        computeElementCount(shape),
        shape.data(),
        shape.size());
}

template <typename T>
inline void extractTensorData(const Ort::Value& tensor,
                              std::vector<T>& dataOut,
                              std::vector<int64_t>& shapeOut)
{
    const T* tensorData = tensor.GetTensorData<T>();
    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    const size_t count = computeElementCount(shape);
    dataOut.assign(tensorData, tensorData + count);
    shapeOut.assign(shape.begin(), shape.end());
}

struct SAM3Point {
    int x = 0;
    int y = 0;
    SAM3Point() = default;
    SAM3Point(int xValue, int yValue) : x(xValue), y(yValue) {}
};

struct SAM3Rect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    SAM3Rect() = default;
    SAM3Rect(int xValue, int yValue, int widthValue, int heightValue)
        : x(xValue), y(yValue), width(widthValue), height(heightValue) {}

    SAM3Point br() const { return SAM3Point(x + width, y + height); }
};

enum class SAM3MaskPromptStrategy {
    Box,
    Point,
};

struct SAM3Prompts {
    std::vector<SAM3Point> points;
    std::vector<int> pointLabels;
    std::vector<SAM3Rect> rects;
    Image<float> mask;
    SAM3MaskPromptStrategy maskPromptStrategy = SAM3MaskPromptStrategy::Box;
};

struct SAM3LogitsTensor {
    std::vector<int64_t> shape;
    bool empty() const noexcept { return shape.empty(); }
};

struct SAM3MaskCandidates {
    Image<float> selectedMask;
    std::vector<Image<float>> masks;
    std::vector<float> scores;
    int selectedIndex = -1;
    SAM3LogitsTensor selectedMaskLogitsHighRes;
    SAM3LogitsTensor multimaskLogitsHighRes;

    bool hasCandidates() const { return !masks.empty(); }
    bool hasSelectedLogits() const { return !selectedMaskLogitsHighRes.empty(); }
    bool hasMultimaskLogits() const { return !multimaskLogitsHighRes.empty(); }
};

struct SAM3MaskSelection {
    int candidateIndex = -1;
    Image<float> overrideMask;

    bool hasOverrideMask() const
    {
        return overrideMask.getWidth() > 0
            && overrideMask.getHeight() > 0
            && overrideMask.getChannels() == 1;
    }
};

using TrackerMaskSelectionCallback =
    std::function<SAM3MaskSelection(const SAM3MaskCandidates&)>;

struct PreparedSAM3MaskPrompt {
    std::vector<float> maskLogitsHighRes;
    std::vector<int64_t> maskLogitsShape;
    std::vector<float> maskPrompt;
    std::vector<int64_t> maskPromptShape;
    Image<float> originalMask;
    std::vector<float> fallbackPointCoords;
    std::vector<int32_t> fallbackPointLabels;
};

struct SAM3Node {
    std::string name;
    std::vector<int64_t> dim;
};

struct SAM3Size {
    int width = 0;
    int height = 0;
    SAM3Size() = default;
    SAM3Size(int widthValue, int heightValue) : width(widthValue), height(heightValue) {}
};

enum class CachedTensorElementType : std::uint8_t {
    Float32,
    Float16
};

struct CachedTensorData {
    std::vector<float> values;
    std::vector<std::uint16_t> halfValues;
    std::vector<int64_t> shape;
    CachedTensorElementType elementType = CachedTensorElementType::Float32;

    bool empty() const noexcept
    {
        return elementType == CachedTensorElementType::Float16
            ? halfValues.empty()
            : values.empty();
    }

    std::size_t storedBytes() const noexcept
    {
        return values.size() * sizeof(float)
            + halfValues.size() * sizeof(std::uint16_t)
            + shape.size() * sizeof(int64_t);
    }
};

struct CachedEncoderOutputs {
    CachedTensorData imageEmb0;
    CachedTensorData imageEmb1;
    CachedTensorData imageEmb2;
};

struct SAM3Constants {
    std::vector<float> noMemEmbed;
    std::vector<int64_t> noMemEmbedShape;
    std::vector<float> currentVisionPosEmbed;
    std::vector<int64_t> currentVisionPosEmbedShape;
    int numMaskmem = 7;
    int maxObjPtrs = 16;
    int maxCondFramesInAttn = 4;
    bool keepFirstCondFrame = false;
    int memoryTemporalStrideForEval = 1;
    bool useMemorySelection = false;
    float mfThreshold = 0.01f;
    int exportMaxMemFrames = 2;
    int exportMaxObjPtrs = 16;
};

struct TrackerFrameState {
    std::vector<float> maskmemFeatures;
    std::vector<float> maskmemPosEnc;
    std::vector<float> objPtr;
    int frameIndex = -1;
    float objectScoreLogit = 0.0f;
    float effIouScore = 0.0f;
    bool hasEffIouScore = false;
};

// Scalar-only view for quality gates and diagnostics. Callers that only need
// confidence must not copy the multi-megabyte memory feature tensors.
struct TrackerFrameMetrics {
    int frameIndex = -1;
    float objectScoreLogit = 0.0f;
    float effIouScore = 0.0f;
    bool hasEffIouScore = false;
};

using TrackerFrameStateHandle = std::shared_ptr<const TrackerFrameState>;

struct SAM3MemorySnapshot {
    bool hasConditioningState = false;
    std::deque<TrackerFrameStateHandle> conditioningStates;
    std::deque<TrackerFrameStateHandle> nonConditioningStates;
    TrackerFrameStateHandle lastTrackerFrameState;
    bool hasLastTrackerFrameState = false;
    int segmentFrameIndex = 0;
};

struct SAM3FrameTimings {
    int frameIndex = -1;
    bool conditioningFrame = false;
    bool denseMaskConditioning = false;
    bool usedSelectionCallback = false;
    int candidateCount = 0;
    double totalMs = 0.0;
    double encMs = 0.0;
    double maskPrepMs = 0.0;
    double promptBuildMs = 0.0;
    double noMemoryEmbedMs = 0.0;
    double memoryBuildMs = 0.0;
    double attnMs = 0.0;
    double decoderMs = 0.0;
    double decMs = 0.0;
    double candidateMs = 0.0;
    double selectionMs = 0.0;
    double selectionLogitsMs = 0.0;
    double memMs = 0.0;
    double captureStateMs = 0.0;
    double stateUpdateMs = 0.0;
    bool propagationIoBindingRequested = false;
    bool propagationIoBindingUsed = false;
    bool propagationIoBindingFellBack = false;
};

struct SAM3DiagnosticsOptions {
    bool runtimeMetadata = false;
    bool frameTimings = false;
};

// Session roles get independent graph-optimization policies. The encoder is the
// large, well-behaved community graph; the tracker graphs are the ones that are
// sensitive to aggressive ORT rewrites (see README "safe mode" notes).
enum class SAM3SessionRole {
    Encoder,
    Tracker,
};

struct SAM3RuntimeMetadata {
    std::string mode;
    std::string device;
    SAM3Size inputSize;
    bool videoInitialized = false;
    bool imageInitialized = false;
    bool hasVideoConstants = false;
    bool decoderHasMultimasks = false;
    bool decoderHasIouScores = false;
    int trackerDecoderObjPtrIndex = -1;
    int trackerDecoderPredMaskHighResIndex = -1;
    int trackerDecoderPredMultimasksHighResIndex = -1;
    int trackerDecoderObjectScoreIndex = -1;
    int trackerDecoderIouScoresIndex = -1;
    int memoryAttentionFusedFeatIndex = -1;
    int memoryEncoderFeaturesIndex = -1;
    int memoryEncoderPosEncIndex = -1;
    int staticNumMemFrames = 0;
    int staticNumObjPtrs = 0;
    int effectiveMaxMemFrames = 0;
    int effectiveMaxObjPtrs = 0;
    int maxCondFramesInAttn = 0;
    bool keepFirstCondFrame = false;
    int memoryTemporalStrideForEval = 1;
    bool useMemorySelection = false;
    float mfThreshold = 0.0f;
    std::vector<std::string> trackerDecoderOutputNames;
    std::vector<SAM3Node> encoderInputNodes;
    std::vector<SAM3Node> encoderOutputNodes;
    std::vector<SAM3Node> trackerDecoderInputNodes;
    std::vector<SAM3Node> trackerDecoderOutputNodes;
    std::vector<SAM3Node> memoryAttentionInputNodes;
    std::vector<SAM3Node> memoryAttentionOutputNodes;
    std::vector<SAM3Node> memoryEncoderInputNodes;
    std::vector<SAM3Node> memoryEncoderOutputNodes;
    std::vector<SAM3Node> imageDecoderInputNodes;
    std::vector<SAM3Node> imageDecoderOutputNodes;
};

class SAM3 {
public:
    SAM3();
    ~SAM3();

    bool initializeImage(const std::string& encoderPath,
                         const std::string& decoderPath,
                         int threadsNumber,
                         const std::string& device = "cpu");
    bool initializeImage(const std::string& encoderPath,
                         const std::string& decoderPath,
                         const std::string& constantsPath,
                         int threadsNumber,
                         const std::string& device = "cpu");

    bool initializeVideo(const std::string& encoderPath,
                         const std::string& decoderPath,
                         const std::string& memoryAttentionPath,
                         const std::string& memoryEncoderPath,
                         const std::string& constantsPath,
                         int threadsNumber,
                         const std::string& device = "cpu");

    bool preprocessImage(const Image<float>& originalImage);
    bool captureCachedEncoderOutputs(CachedEncoderOutputs* outputs) const;
    bool restoreCachedEncoderOutputs(const CachedEncoderOutputs& outputs);

    Image<float> inferSingleFrame(const SAM3Size& originalImageSize,
                                  const SAM3Prompts& prompts);
    Image<float> previewConditioningFrame(const SAM3Size& originalImageSize,
                                          const SAM3Prompts& prompts);
    SAM3MaskCandidates previewConditioningFrameCandidates(const SAM3Size& originalImageSize,
                                                          const SAM3Prompts& prompts);
    Image<float> inferMultiFrame(const Image<float>& originalImage,
                                 const SAM3Prompts& prompts);
    Image<float> inferMultiFrameCached(const SAM3Size& originalImageSize,
                                       const SAM3Prompts& prompts);

    void setTrackerMaskSelectionCallback(TrackerMaskSelectionCallback callback);
    bool lastFrameTimings(SAM3FrameTimings* timingsOut) const;
    bool lastTrackerFrameState(TrackerFrameState* stateOut) const;
    bool lastTrackerFrameMetrics(TrackerFrameMetrics* metricsOut) const;
    bool lastTrackerMaskCandidates(SAM3MaskCandidates* candidatesOut) const;
    void setDiagnosticsOptions(const SAM3DiagnosticsOptions&) {}
    bool runtimeMetadata(SAM3RuntimeMetadata*) const { return false; }
    bool captureMemorySnapshot(SAM3MemorySnapshot* snapshotOut) const;
    void restoreMemorySnapshot(const SAM3MemorySnapshot& snapshot);
    void resetMemory();
    SAM3Size getInputSize() const;
    bool modelExists(const std::string& modelPath) const;

    static bool hasCudaDriver();
    static bool hasDirectMLProvider();
    static void setupSessionOptions(Ort::SessionOptions& options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string& device);
    static void setupSessionOptions(Ort::SessionOptions& options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string& device,
                                    SAM3SessionRole role);
    static std::vector<SAM3Node> getSessionNodes(Ort::Session* session, bool isInput);

private:
    bool clearSessions();
    void warmupVideoRuntime(bool includeEncoder);
    bool initializeNamedSession(std::unique_ptr<Ort::Session>* sessionOut,
                                const Ort::Env& env,
                                const std::string& modelPath,
                                const Ort::SessionOptions& options,
                                std::vector<SAM3Node>* inputNodes,
                                std::vector<SAM3Node>* outputNodes,
                                std::vector<const char*>* inputNames,
                                std::vector<const char*>* outputNames);
    bool loadVideoConstants(const std::string& constantsPath);

    static std::vector<const char*> collectNodeNames(const std::vector<SAM3Node>& nodes);
    static int findNodeIndex(const std::vector<SAM3Node>& nodes, const std::string& key);
    static int findNameIndex(const std::vector<const char*>& names, const std::string& key);
    static std::string lowerCopy(const std::string& value);

    std::variant<std::vector<Ort::Value>, std::string> runSession(
        Ort::Session* session,
        const std::vector<const char*>& inputNames,
        const std::vector<const char*>& outputNames,
        const std::vector<Ort::Value>& inputTensors,
        const std::string& debugName);
    std::variant<std::vector<Ort::Value>, std::string> runSessionWithOutputMemory(
        Ort::Session* session,
        const std::vector<const char*>& inputNames,
        const std::vector<const char*>& outputNames,
        const std::vector<Ort::Value>& inputTensors,
        const Ort::MemoryInfo& outputMemoryInfo,
        const std::string& debugName);

    const std::vector<float>& buildNoMemoryImageEmbedding(
        const Ort::Value& currentVisionFeat,
        const std::vector<int64_t>& currentVisionShape);
    void invalidateNoMemoryImageEmbeddingCache();
    void buildImagePromptInputs(const SAM3Prompts& prompts,
                                const SAM3Size& originalImageSize,
                                std::vector<float>* pointsOut,
                                std::vector<int64_t>* labelsOut,
                                std::vector<float>* boxesOut) const;
    void buildTrackerPromptInputs(const SAM3Prompts& prompts,
                                  const SAM3Size& originalImageSize,
                                  std::vector<float>* coordsOut,
                                  std::vector<int32_t>* labelsOut) const;
    Image<float> createImageMaskFromLogits(const float* logits,
                                           int maskWidth,
                                           int maskHeight,
                                           const SAM3Size& originalImageSize) const;
    Image<float> createImageMaskFromDecoderOutput(const Ort::Value& predMasks,
                                                  const Ort::Value& iouScores,
                                                  const SAM3Size& originalImageSize) const;
    Image<float> createTrackerMaskFromHighRes(const Ort::Value& predMaskHighRes,
                                              const SAM3Size& originalImageSize) const;
    bool hasMaskPrompt(const SAM3Prompts& prompts) const;
    bool promptsEmpty(const SAM3Prompts& prompts) const;
    bool prepareMaskPrompt(const SAM3Prompts& prompts,
                           const SAM3Size& originalImageSize,
                           PreparedSAM3MaskPrompt* preparedOut) const;
    SAM3MaskCandidates collectTrackerMaskCandidates(const std::vector<Ort::Value>& decoderOutputs,
                                                    const SAM3Size& originalImageSize) const;

    Image<float> inferMultiFrameWithEncoderOutputs(std::vector<Ort::Value>& encoderOutputs,
                                                   const SAM3Size& originalImageSize,
                                                   const SAM3Prompts& prompts,
                                                   double encTimeMs);
    TrackerFrameState captureTrackerState(const std::vector<Ort::Value>& decoderOutputs,
                                          const std::vector<Ort::Value>& memoryEncoderOutputs,
                                          int frameIndex);
    void appendNonConditioningState(TrackerFrameStateHandle state);
    void trimNonConditioningStates();
    void buildMemoryInputBuffers(int frameIndex);

private:
    // Single shared environment for all sessions. Declared before the sessions
    // so it outlives them during destruction.
    Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "smseg_sam3"};

    std::unique_ptr<Ort::Session> m_encoderSession;
    std::unique_ptr<Ort::Session> m_imageDecoderSession;
    std::unique_ptr<Ort::Session> m_imageMaskDecoderSession;
    std::unique_ptr<Ort::Session> m_trackerDecoderSession;
    std::unique_ptr<Ort::Session> m_trackerMaskDecoderSession;
    std::unique_ptr<Ort::Session> m_memoryAttentionSession;
    std::unique_ptr<Ort::Session> m_memoryEncoderSession;

    std::vector<SAM3Node> m_encoderInputNodes;
    std::vector<SAM3Node> m_encoderOutputNodes;
    std::vector<SAM3Node> m_imageDecoderInputNodes;
    std::vector<SAM3Node> m_imageDecoderOutputNodes;
    std::vector<SAM3Node> m_imageMaskDecoderInputNodes;
    std::vector<SAM3Node> m_imageMaskDecoderOutputNodes;
    std::vector<SAM3Node> m_trackerDecoderInputNodes;
    std::vector<SAM3Node> m_trackerDecoderOutputNodes;
    std::vector<SAM3Node> m_trackerMaskDecoderInputNodes;
    std::vector<SAM3Node> m_trackerMaskDecoderOutputNodes;
    std::vector<SAM3Node> m_memoryAttentionInputNodes;
    std::vector<SAM3Node> m_memoryAttentionOutputNodes;
    std::vector<SAM3Node> m_memoryEncoderInputNodes;
    std::vector<SAM3Node> m_memoryEncoderOutputNodes;

    std::vector<const char*> m_encoderInputNames;
    std::vector<const char*> m_encoderOutputNames;
    std::vector<const char*> m_imageDecoderInputNames;
    std::vector<const char*> m_imageDecoderOutputNames;
    std::vector<const char*> m_imageMaskDecoderInputNames;
    std::vector<const char*> m_imageMaskDecoderOutputNames;
    std::vector<const char*> m_trackerDecoderInputNames;
    std::vector<const char*> m_trackerDecoderOutputNames;
    std::vector<const char*> m_trackerMaskDecoderInputNames;
    std::vector<const char*> m_trackerMaskDecoderOutputNames;
    std::vector<const char*> m_memoryAttentionInputNames;
    std::vector<const char*> m_memoryAttentionOutputNames;
    std::vector<const char*> m_memoryEncoderInputNames;
    std::vector<const char*> m_memoryEncoderOutputNames;

    std::vector<int64_t> m_inputShapeEncoder;

    int m_encoderImageEmb0Index = -1;
    int m_encoderImageEmb1Index = -1;
    int m_encoderImageEmb2Index = -1;

    int m_imageDecoderPredMasksIndex = -1;
    int m_imageDecoderIouScoresIndex = -1;
    int m_imageDecoderMaskInputIndex = -1;
    bool m_imageDecoderUsesTrackerIo = false;
    int m_imageMaskDecoderPredMasksIndex = -1;
    int m_imageMaskDecoderIouScoresIndex = -1;
    int m_imageMaskDecoderMaskInputIndex = -1;
    bool m_imageMaskDecoderUsesTrackerIo = false;

    int m_trackerDecoderObjPtrIndex = -1;
    int m_trackerDecoderPredMaskHighResIndex = -1;
    int m_trackerDecoderPredMultimasksIndex = -1;
    int m_trackerDecoderPredMultimasksHighResIndex = -1;
    int m_trackerDecoderObjectScoreIndex = -1;
    int m_trackerDecoderIouScoresIndex = -1;
    int m_trackerMaskDecoderObjPtrIndex = -1;
    int m_trackerMaskDecoderObjectScoreIndex = -1;
    int m_trackerMaskDecoderIouScoresIndex = -1;

    int m_memoryAttentionFusedFeatIndex = -1;
    int m_memoryEncoderFeaturesIndex = -1;
    int m_memoryEncoderPosIndex = -1;

    int m_staticNumMemFrames = 0;
    int m_staticNumObjPtrs = 0;

    std::vector<Ort::Value> m_cachedEncoderOutputs;
    CachedEncoderOutputs m_cachedEncoderHostCopy;
    bool m_hasCachedEncoderHostCopy = false;
    bool m_compressEncoderCacheToHalf = false;

    SAM3Constants m_constants;
    bool m_hasVideoConstants = false;
    bool m_hasConditioningState = false;
    std::deque<TrackerFrameStateHandle> m_conditioningStates;
    std::deque<TrackerFrameStateHandle> m_nonConditioningStates;
    TrackerFrameStateHandle m_lastTrackerFrameState;
    bool m_hasLastTrackerFrameState = false;
    SAM3MaskCandidates m_lastTrackerMaskCandidates;
    bool m_hasLastTrackerMaskCandidates = false;
    SAM3FrameTimings m_lastFrameTimings;
    bool m_hasLastFrameTimings = false;
    TrackerMaskSelectionCallback m_trackerMaskSelectionCallback;
    int m_segmentFrameIndex = 0;

    std::vector<float> m_noMemoryImageEmbedScratch;
    std::vector<int64_t> m_noMemoryImageEmbedCacheShape;
    const float* m_noMemoryImageEmbedCacheSource = nullptr;
    bool m_hasNoMemoryImageEmbedCache = false;
    std::vector<float> m_memoryObjPtrsScratch;
    std::vector<float> m_memoryObjTposScratch;
    std::vector<float> m_memoryMaskFeatsScratch;
    std::vector<float> m_memoryMaskPosScratch;
    std::vector<int64_t> m_memoryMaskTposScratch;

    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo m_cudaMemoryInfo{nullptr};
    bool m_usePropagationIoBinding = false;
    bool m_propagationIoBindingDisabledAfterFailure = false;
    std::string m_device = "cpu";
};

} // namespace smseg_sam3

#endif // SAM3CPP__SAM3_H_
