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
#include <initializer_list>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "Image.h"

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

struct PreparedSAM3MaskPrompt {
    std::vector<float> maskLogitsHighRes;
    std::vector<int64_t> maskLogitsShape;
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

struct SAM3MaskCandidates {
    Image<float> selectedMask;
    std::vector<Image<float>> masks;
    std::vector<float> scores;
    int selectedIndex = -1;
    CachedTensorData selectedMaskLogitsHighRes;
    CachedTensorData multimaskLogitsHighRes;

    bool hasCandidates() const { return !masks.empty(); }
    bool hasSelectedLogits() const { return !selectedMaskLogitsHighRes.values.empty(); }
    bool hasMultimaskLogits() const { return !multimaskLogitsHighRes.values.empty(); }
};

struct SAM3DiagnosticsOptions {
    bool captureTrackerCandidates = false;
    bool captureRawTrackerLogits = false;
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

struct SAM3MemorySnapshot {
    bool hasConditioningState = false;
    TrackerFrameState conditioningState;
    std::deque<TrackerFrameState> nonConditioningStates;
    int segmentFrameIndex = 0;
};

struct SAM3FrameTimings {
    int frameIndex = -1;
    bool conditioningFrame = false;
    double encMs = 0.0;
    double attnMs = 0.0;
    double decMs = 0.0;
    double memMs = 0.0;
    double totalMs = 0.0;
};

struct SAM3RuntimeMetadata {
    std::string mode;
    std::string device;
    std::string encoderPath;
    std::string imageDecoderPath;
    std::string trackerDecoderPath;
    std::string memoryAttentionPath;
    std::string memoryEncoderPath;
    std::string constantsPath;
    SAM3Size inputSize;
    bool imageInitialized = false;
    bool videoInitialized = false;
    bool hasVideoConstants = false;
    bool decoderHasIouScores = false;
    bool decoderHasMultimasks = false;
    int imageDecoderPredMasksIndex = -1;
    int imageDecoderIouScoresIndex = -1;
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
    int memoryTemporalStrideForEval = 0;
    bool useMemorySelection = false;
    float mfThreshold = 0.0f;
    int exportMaxMemFrames = 0;
    int exportMaxObjPtrs = 0;
    size_t nonConditioningFramesKept = 0;
    int segmentFrameIndex = 0;
    std::vector<SAM3Node> encoderInputNodes;
    std::vector<SAM3Node> encoderOutputNodes;
    std::vector<SAM3Node> imageDecoderInputNodes;
    std::vector<SAM3Node> imageDecoderOutputNodes;
    std::vector<SAM3Node> trackerDecoderInputNodes;
    std::vector<SAM3Node> trackerDecoderOutputNodes;
    std::vector<SAM3Node> memoryAttentionInputNodes;
    std::vector<SAM3Node> memoryAttentionOutputNodes;
    std::vector<SAM3Node> memoryEncoderInputNodes;
    std::vector<SAM3Node> memoryEncoderOutputNodes;
    std::vector<std::string> encoderInputNames;
    std::vector<std::string> encoderOutputNames;
    std::vector<std::string> imageDecoderInputNames;
    std::vector<std::string> imageDecoderOutputNames;
    std::vector<std::string> trackerDecoderInputNames;
    std::vector<std::string> trackerDecoderOutputNames;
    std::vector<std::string> memoryAttentionInputNames;
    std::vector<std::string> memoryAttentionOutputNames;
    std::vector<std::string> memoryEncoderInputNames;
    std::vector<std::string> memoryEncoderOutputNames;
};

class SAM3 {
public:
    SAM3();
    ~SAM3();

    bool initializeImage(const std::string& encoderPath,
                         const std::string& decoderPath,
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
    bool preprocessImageTensor(const float* encoderInputData, size_t elementCount);
    bool preprocessImageTensor(const std::vector<float>& encoderInputData);
    bool captureCachedEncoderOutputs(CachedEncoderOutputs* outputs) const;
    bool restoreCachedEncoderOutputs(const CachedEncoderOutputs& outputs);

    Image<float> inferSingleFrame(const SAM3Size& originalImageSize,
                                  const SAM3Prompts& prompts);
    Image<float> inferSingleFrameTensor(const float* encoderInputData,
                                        size_t elementCount,
                                        const SAM3Size& originalImageSize,
                                        const SAM3Prompts& prompts);
    Image<float> inferSingleFrameTensor(const std::vector<float>& encoderInputData,
                                        const SAM3Size& originalImageSize,
                                        const SAM3Prompts& prompts);
    Image<float> previewConditioningFrame(const SAM3Size& originalImageSize,
                                          const SAM3Prompts& prompts);
    SAM3MaskCandidates previewConditioningFrameCandidates(const SAM3Size& originalImageSize,
                                                          const SAM3Prompts& prompts);
    Image<float> inferMultiFrame(const Image<float>& originalImage,
                                 const SAM3Prompts& prompts);
    Image<float> inferMultiFrameTensor(const float* encoderInputData,
                                       size_t elementCount,
                                       const SAM3Size& originalImageSize,
                                       const SAM3Prompts& prompts);
    Image<float> inferMultiFrameTensor(const std::vector<float>& encoderInputData,
                                       const SAM3Size& originalImageSize,
                                       const SAM3Prompts& prompts);
    Image<float> inferMultiFrameCached(const SAM3Size& originalImageSize,
                                       const SAM3Prompts& prompts);

    void setDiagnosticsOptions(const SAM3DiagnosticsOptions& options);
    SAM3DiagnosticsOptions diagnosticsOptions() const;
    bool runtimeMetadata(SAM3RuntimeMetadata* metadataOut) const;
    bool lastFrameTimings(SAM3FrameTimings* timingsOut) const;
    bool lastTrackerFrameState(TrackerFrameState* stateOut) const;
    bool lastTrackerMaskCandidates(SAM3MaskCandidates* candidatesOut) const;
    void resetMemory();
    bool captureMemorySnapshot(SAM3MemorySnapshot* snapshotOut) const;
    void restoreMemorySnapshot(const SAM3MemorySnapshot& snapshot);
    SAM3Size getInputSize() const;
    bool modelExists(const std::string& modelPath) const;

    static bool hasCudaDriver();
    static bool hasDirectMLProvider();
    static void setupSessionOptions(Ort::SessionOptions& options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string& device);
    static std::vector<SAM3Node> getSessionNodes(Ort::Session* session, bool isInput);

private:
    bool clearSessions();
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

    const std::vector<float>& buildNoMemoryImageEmbedding(
        const Ort::Value& currentVisionFeat,
        const std::vector<int64_t>& currentVisionShape);
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
    Image<float> createTrackerMaskFromHighRes(const Ort::Value& predMaskHighRes,
                                              const SAM3Size& originalImageSize) const;
    bool hasMaskPrompt(const SAM3Prompts& prompts) const;
    bool promptsEmpty(const SAM3Prompts& prompts) const;
    bool prepareMaskPrompt(const SAM3Prompts& prompts,
                           const SAM3Size& originalImageSize,
                           PreparedSAM3MaskPrompt* preparedOut) const;
    SAM3MaskCandidates collectTrackerMaskCandidates(const std::vector<Ort::Value>& decoderOutputs,
                                                    const SAM3Size& originalImageSize,
                                                    bool includeRawLogits) const;

    Image<float> inferMultiFrameWithEncoderOutputs(std::vector<Ort::Value>& encoderOutputs,
                                                   const SAM3Size& originalImageSize,
                                                   const SAM3Prompts& prompts,
                                                   double encTimeMs);
    TrackerFrameState captureTrackerState(const std::vector<Ort::Value>& decoderOutputs,
                                          const std::vector<Ort::Value>& memoryEncoderOutputs,
                                          int frameIndex);
    void appendNonConditioningState(const TrackerFrameState& state);
    void trimNonConditioningStates();
    void buildMemoryInputBuffers(int frameIndex);

private:
    std::unique_ptr<Ort::Session> m_encoderSession;
    std::unique_ptr<Ort::Session> m_imageDecoderSession;
    std::unique_ptr<Ort::Session> m_trackerDecoderSession;
    std::unique_ptr<Ort::Session> m_memoryAttentionSession;
    std::unique_ptr<Ort::Session> m_memoryEncoderSession;

    std::vector<SAM3Node> m_encoderInputNodes;
    std::vector<SAM3Node> m_encoderOutputNodes;
    std::vector<SAM3Node> m_imageDecoderInputNodes;
    std::vector<SAM3Node> m_imageDecoderOutputNodes;
    std::vector<SAM3Node> m_trackerDecoderInputNodes;
    std::vector<SAM3Node> m_trackerDecoderOutputNodes;
    std::vector<SAM3Node> m_memoryAttentionInputNodes;
    std::vector<SAM3Node> m_memoryAttentionOutputNodes;
    std::vector<SAM3Node> m_memoryEncoderInputNodes;
    std::vector<SAM3Node> m_memoryEncoderOutputNodes;

    std::vector<const char*> m_encoderInputNames;
    std::vector<const char*> m_encoderOutputNames;
    std::vector<const char*> m_imageDecoderInputNames;
    std::vector<const char*> m_imageDecoderOutputNames;
    std::vector<const char*> m_trackerDecoderInputNames;
    std::vector<const char*> m_trackerDecoderOutputNames;
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

    int m_trackerDecoderObjPtrIndex = -1;
    int m_trackerDecoderPredMaskHighResIndex = -1;
    int m_trackerDecoderPredMultimasksHighResIndex = -1;
    int m_trackerDecoderObjectScoreIndex = -1;
    int m_trackerDecoderIouScoresIndex = -1;

    int m_memoryAttentionFusedFeatIndex = -1;
    int m_memoryEncoderFeaturesIndex = -1;
    int m_memoryEncoderPosIndex = -1;

    int m_staticNumMemFrames = 0;
    int m_staticNumObjPtrs = 0;

    std::vector<Ort::Value> m_cachedEncoderOutputs;
    CachedEncoderOutputs m_cachedEncoderHostCopy;
    bool m_hasCachedEncoderHostCopy = false;

    SAM3Constants m_constants;
    bool m_hasVideoConstants = false;
    bool m_hasConditioningState = false;
    TrackerFrameState m_conditioningState;
    std::deque<TrackerFrameState> m_nonConditioningStates;
    int m_segmentFrameIndex = 0;
    SAM3DiagnosticsOptions m_diagnosticsOptions;
    SAM3FrameTimings m_lastFrameTimings;
    bool m_hasLastFrameTimings = false;
    TrackerFrameState m_lastTrackerFrameState;
    bool m_hasLastTrackerFrameState = false;
    SAM3MaskCandidates m_lastTrackerMaskCandidates;
    bool m_hasLastTrackerMaskCandidates = false;

    std::vector<float> m_noMemoryImageEmbedScratch;
    std::vector<float> m_memoryObjPtrsScratch;
    std::vector<float> m_memoryObjTposScratch;
    std::vector<float> m_memoryMaskFeatsScratch;
    std::vector<float> m_memoryMaskPosScratch;
    std::vector<int64_t> m_memoryMaskTposScratch;

    Ort::Env m_encoderEnv{ORT_LOGGING_LEVEL_WARNING, "sam3_encoder"};
    Ort::Env m_imageDecoderEnv{ORT_LOGGING_LEVEL_WARNING, "sam3_image_decoder"};
    Ort::Env m_trackerDecoderEnv{ORT_LOGGING_LEVEL_WARNING, "sam3_tracker_decoder"};
    Ort::Env m_memoryAttentionEnv{ORT_LOGGING_LEVEL_WARNING, "sam3_memory_attention"};
    Ort::Env m_memoryEncoderEnv{ORT_LOGGING_LEVEL_WARNING, "sam3_memory_encoder"};

    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string m_device = "cpu";
    std::string m_encoderPath;
    std::string m_imageDecoderPath;
    std::string m_trackerDecoderPath;
    std::string m_memoryAttentionPath;
    std::string m_memoryEncoderPath;
    std::string m_constantsPath;
};

#endif // SAM3CPP__SAM3_H_
