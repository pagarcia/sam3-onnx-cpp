#include "SAM3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "CVHelpers.h"

namespace {

size_t safeMinCount(size_t a, size_t b)
{
    return std::min(a, b);
}

} // namespace

void SAM3::buildTrackerPromptInputs(const SAM3Prompts& prompts,
                                    const SAM3Size& originalImageSize,
                                    std::vector<float>* coordsOut,
                                    std::vector<int32_t>* labelsOut) const
{
    coordsOut->clear();
    labelsOut->clear();

    const SAM3Size inputSize = getInputSize();
    if (inputSize.width <= 0 || inputSize.height <= 0
        || originalImageSize.width <= 0 || originalImageSize.height <= 0) {
        return;
    }

    for (const auto& rawRect : prompts.rects) {
        const int x1 = std::min(rawRect.x, rawRect.x + rawRect.width);
        const int y1 = std::min(rawRect.y, rawRect.y + rawRect.height);
        const int x2 = std::max(rawRect.x, rawRect.x + rawRect.width);
        const int y2 = std::max(rawRect.y, rawRect.y + rawRect.height);

        coordsOut->push_back(x1 * static_cast<float>(inputSize.width) / originalImageSize.width);
        coordsOut->push_back(y1 * static_cast<float>(inputSize.height) / originalImageSize.height);
        labelsOut->push_back(2);

        coordsOut->push_back(x2 * static_cast<float>(inputSize.width) / originalImageSize.width);
        coordsOut->push_back(y2 * static_cast<float>(inputSize.height) / originalImageSize.height);
        labelsOut->push_back(3);
    }

    const size_t pointCount = std::min(prompts.points.size(), prompts.pointLabels.size());
    for (size_t index = 0; index < pointCount; ++index) {
        coordsOut->push_back(
            prompts.points[index].x * static_cast<float>(inputSize.width) / originalImageSize.width);
        coordsOut->push_back(
            prompts.points[index].y * static_cast<float>(inputSize.height) / originalImageSize.height);
        labelsOut->push_back(static_cast<int32_t>(prompts.pointLabels[index]));
    }
}

Image<float> SAM3::createTrackerMaskFromHighRes(const Ort::Value& predMaskHighRes,
                                                const SAM3Size& originalImageSize) const
{
    const auto shape = predMaskHighRes.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() < 4) {
        return Image<float>();
    }

    const float* maskData = predMaskHighRes.GetTensorData<float>();
    return CVHelpers::resizeAndThresholdMask(
        maskData,
        static_cast<int>(shape[3]),
        static_cast<int>(shape[2]),
        originalImageSize.width,
        originalImageSize.height,
        0.0f);
}

TrackerFrameState SAM3::captureTrackerState(const std::vector<Ort::Value>& decoderOutputs,
                                            const std::vector<Ort::Value>& memoryEncoderOutputs,
                                            int frameIndex)
{
    TrackerFrameState state;
    state.frameIndex = frameIndex;

    if (memoryEncoderOutputs.size() <= static_cast<size_t>(std::max(m_memoryEncoderFeaturesIndex, m_memoryEncoderPosIndex))
        || decoderOutputs.size() <= static_cast<size_t>(m_trackerDecoderObjPtrIndex)) {
        throw std::runtime_error("Tracker state outputs are missing required tensors.");
    }

    std::vector<int64_t> scratchShape;
    extractTensorData(memoryEncoderOutputs[static_cast<size_t>(m_memoryEncoderFeaturesIndex)], state.maskmemFeatures, scratchShape);
    extractTensorData(memoryEncoderOutputs[static_cast<size_t>(m_memoryEncoderPosIndex)], state.maskmemPosEnc, scratchShape);
    extractTensorData(decoderOutputs[static_cast<size_t>(m_trackerDecoderObjPtrIndex)], state.objPtr, scratchShape);

    if (m_trackerDecoderObjectScoreIndex >= 0
        && decoderOutputs.size() > static_cast<size_t>(m_trackerDecoderObjectScoreIndex)) {
        const float* logits = decoderOutputs[static_cast<size_t>(m_trackerDecoderObjectScoreIndex)].GetTensorData<float>();
        if (logits) {
            state.objectScoreLogit = logits[0];
        }
    }

    if (m_trackerDecoderIouScoresIndex >= 0
        && decoderOutputs.size() > static_cast<size_t>(m_trackerDecoderIouScoresIndex)) {
        const Ort::Value& iouTensor = decoderOutputs[static_cast<size_t>(m_trackerDecoderIouScoresIndex)];
        const auto iouShape = iouTensor.GetTensorTypeAndShapeInfo().GetShape();
        const float* iouValues = iouTensor.GetTensorData<float>();
        const size_t iouCount = computeElementCount(iouShape);
        if (iouValues && iouCount > 0) {
            const float bestIou = *std::max_element(iouValues, iouValues + iouCount);
            if (state.objectScoreLogit > 0.0f) {
                const float sigmoid = 1.0f / (1.0f + std::exp(-state.objectScoreLogit));
                state.effIouScore = (sigmoid * 2.0f - 1.0f) * bestIou;
                state.hasEffIouScore = true;
            }
        }
    }

    return state;
}

void SAM3::appendNonConditioningState(const TrackerFrameState& state)
{
    m_nonConditioningStates.push_back(state);
    trimNonConditioningStates();
}

void SAM3::trimNonConditioningStates()
{
    const size_t keepCount = static_cast<size_t>(std::max(
        std::max(m_staticNumObjPtrs, m_staticNumMemFrames),
        std::max(m_constants.maxObjPtrs, m_constants.numMaskmem)));
    while (m_nonConditioningStates.size() > keepCount) {
        m_nonConditioningStates.pop_front();
    }
}

void SAM3::buildMemoryInputBuffers(int frameIndex)
{
    const size_t memorySlotCount = static_cast<size_t>(std::max(1, m_staticNumMemFrames));
    const size_t objSlotCount = static_cast<size_t>(std::max(1, m_staticNumObjPtrs));
    const size_t featurePlaneSize = static_cast<size_t>(64 * 72 * 72);
    const size_t objPtrSize = 256;

    m_memoryMaskFeatsScratch.assign(memorySlotCount * featurePlaneSize, 0.0f);
    m_memoryMaskPosScratch.assign(memorySlotCount * featurePlaneSize, 0.0f);
    m_memoryMaskTposScratch.assign(memorySlotCount, 0);
    m_memoryObjPtrsScratch.assign(objSlotCount * objPtrSize, 0.0f);
    m_memoryObjTposScratch.assign(objSlotCount, 0.0f);

    size_t memoryRow = 0;
    if (m_hasConditioningState && memoryRow < memorySlotCount) {
        const size_t copyCount = safeMinCount(featurePlaneSize, m_conditioningState.maskmemFeatures.size());
        const size_t posCopyCount = safeMinCount(featurePlaneSize, m_conditioningState.maskmemPosEnc.size());
        std::copy_n(
            m_conditioningState.maskmemFeatures.begin(),
            copyCount,
            m_memoryMaskFeatsScratch.begin() + static_cast<ptrdiff_t>(memoryRow * featurePlaneSize));
        std::copy_n(
            m_conditioningState.maskmemPosEnc.begin(),
            posCopyCount,
            m_memoryMaskPosScratch.begin() + static_cast<ptrdiff_t>(memoryRow * featurePlaneSize));
        m_memoryMaskTposScratch[memoryRow] = std::max(0, m_constants.numMaskmem - 1);
        ++memoryRow;
    }

    int relativeMemoryIndex = 0;
    for (auto it = m_nonConditioningStates.rbegin();
         it != m_nonConditioningStates.rend() && memoryRow < memorySlotCount;
         ++it, ++relativeMemoryIndex) {
        const size_t copyCount = safeMinCount(featurePlaneSize, it->maskmemFeatures.size());
        const size_t posCopyCount = safeMinCount(featurePlaneSize, it->maskmemPosEnc.size());
        std::copy_n(
            it->maskmemFeatures.begin(),
            copyCount,
            m_memoryMaskFeatsScratch.begin() + static_cast<ptrdiff_t>(memoryRow * featurePlaneSize));
        std::copy_n(
            it->maskmemPosEnc.begin(),
            posCopyCount,
            m_memoryMaskPosScratch.begin() + static_cast<ptrdiff_t>(memoryRow * featurePlaneSize));
        m_memoryMaskTposScratch[memoryRow] = relativeMemoryIndex;
        ++memoryRow;
    }

    size_t objRow = 0;
    if (m_hasConditioningState && objRow < objSlotCount) {
        const size_t copyCount = safeMinCount(objPtrSize, m_conditioningState.objPtr.size());
        std::copy_n(
            m_conditioningState.objPtr.begin(),
            copyCount,
            m_memoryObjPtrsScratch.begin() + static_cast<ptrdiff_t>(objRow * objPtrSize));
        m_memoryObjTposScratch[objRow] =
            static_cast<float>(std::max(0, frameIndex - m_conditioningState.frameIndex));
        ++objRow;
    }

    int relativePointerIndex = 1;
    for (auto it = m_nonConditioningStates.rbegin();
         it != m_nonConditioningStates.rend() && objRow < objSlotCount;
         ++it, ++relativePointerIndex) {
        const size_t copyCount = safeMinCount(objPtrSize, it->objPtr.size());
        std::copy_n(
            it->objPtr.begin(),
            copyCount,
            m_memoryObjPtrsScratch.begin() + static_cast<ptrdiff_t>(objRow * objPtrSize));
        m_memoryObjTposScratch[objRow] = static_cast<float>(relativePointerIndex);
        ++objRow;
    }
}

Image<float> SAM3::previewConditioningFrame(const SAM3Size& originalImageSize,
                                            const SAM3Prompts& prompts)
{
    if (!m_trackerDecoderSession || !m_hasVideoConstants) {
        std::cerr << "[ERROR] previewConditioningFrame => tracker sessions are not initialized.\n";
        return Image<float>();
    }

    const int requiredMaxIndex = std::max({m_encoderImageEmb0Index, m_encoderImageEmb1Index, m_encoderImageEmb2Index});
    if (requiredMaxIndex < 0 || m_cachedEncoderOutputs.size() <= static_cast<size_t>(requiredMaxIndex)) {
        std::cerr << "[ERROR] previewConditioningFrame => encoder outputs are not cached.\n";
        return Image<float>();
    }

    if (promptsEmpty(prompts)) {
        return Image<float>(originalImageSize.width, originalImageSize.height, 1);
    }

    if (hasMaskPrompt(prompts)) {
        PreparedSAM3MaskPrompt preparedMask;
        if (prepareMaskPrompt(prompts, originalImageSize, &preparedMask)) {
            return preparedMask.originalMask;
        }
        return Image<float>(originalImageSize.width, originalImageSize.height, 1);
    }

    try {
        std::vector<float> pointCoords;
        std::vector<int32_t> pointLabels;
        buildTrackerPromptInputs(prompts, originalImageSize, &pointCoords, &pointLabels);
        if (pointLabels.empty()) {
            return Image<float>(originalImageSize.width, originalImageSize.height, 1);
        }

        const std::vector<int64_t> pointCoordsShape = {1, static_cast<int64_t>(pointLabels.size()), 2};
        const std::vector<int64_t> pointLabelsShape = {1, static_cast<int64_t>(pointLabels.size())};
        std::vector<int64_t> currentVisionShape;
        std::vector<float> currentVisionValues;
        extractTensorData(
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)],
            currentVisionValues,
            currentVisionShape);
        const std::vector<float> imageEmbed = buildNoMemoryImageEmbedding(
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)]);

        std::vector<Ort::Value> inputs;
        inputs.reserve(m_trackerDecoderInputNodes.size());
        inputs.push_back(createTensor<float>(m_memoryInfo, pointCoords, pointCoordsShape));
        inputs.push_back(createTensor<int32_t>(m_memoryInfo, pointLabels, pointLabelsShape));
        inputs.push_back(createTensor<float>(m_memoryInfo, imageEmbed, currentVisionShape));
        inputs.push_back(createTensorView<float>(
            m_memoryInfo,
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)].GetTensorMutableData<float>(),
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)].GetTensorTypeAndShapeInfo().GetShape()));
        inputs.push_back(createTensorView<float>(
            m_memoryInfo,
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)].GetTensorMutableData<float>(),
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)].GetTensorTypeAndShapeInfo().GetShape()));

        auto result = runSession(
            m_trackerDecoderSession.get(),
            m_trackerDecoderInputNames,
            m_trackerDecoderOutputNames,
            inputs,
            "trackerPreview");
        if (result.index() == 1) {
            std::cerr << std::get<std::string>(result) << '\n';
            return Image<float>();
        }

        auto outputs = std::move(std::get<0>(result));
        if (outputs.size() <= static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)) {
            std::cerr << "[ERROR] previewConditioningFrame => decoder did not return pred_mask_high_res.\n";
            return Image<float>();
        }
        return createTrackerMaskFromHighRes(
            outputs[static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)],
            originalImageSize);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] previewConditioningFrame => " << error.what() << '\n';
        return Image<float>();
    }
}

Image<float> SAM3::inferMultiFrame(const Image<float>& originalImage,
                                   const SAM3Prompts& prompts)
{
    const SAM3Size originalSize(originalImage.getWidth(), originalImage.getHeight());
    const auto start = std::chrono::steady_clock::now();
    if (!preprocessImage(originalImage)) {
        return Image<float>();
    }
    const double encTimeMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return inferMultiFrameWithEncoderOutputs(m_cachedEncoderOutputs, originalSize, prompts, encTimeMs);
}

Image<float> SAM3::inferMultiFrameCached(const SAM3Size& originalImageSize,
                                         const SAM3Prompts& prompts)
{
    if (m_cachedEncoderOutputs.empty()) {
        std::cerr << "[ERROR] inferMultiFrameCached => encoder outputs are not cached.\n";
        return Image<float>();
    }
    return inferMultiFrameWithEncoderOutputs(m_cachedEncoderOutputs, originalImageSize, prompts, 0.0);
}

Image<float> SAM3::inferMultiFrameWithEncoderOutputs(std::vector<Ort::Value>& encoderOutputs,
                                                     const SAM3Size& originalImageSize,
                                                     const SAM3Prompts& prompts,
                                                     double encTimeMs)
{
    if (!m_trackerDecoderSession || !m_memoryAttentionSession || !m_memoryEncoderSession || !m_hasVideoConstants) {
        std::cerr << "[ERROR] inferMultiFrame => tracker sessions are not initialized.\n";
        return Image<float>();
    }

    const int requiredMaxIndex = std::max({m_encoderImageEmb0Index, m_encoderImageEmb1Index, m_encoderImageEmb2Index});
    if (requiredMaxIndex < 0 || encoderOutputs.size() <= static_cast<size_t>(requiredMaxIndex)) {
        std::cerr << "[ERROR] inferMultiFrame => encoder outputs are missing required tensors.\n";
        return Image<float>();
    }

    const bool conditioningFrame = !promptsEmpty(prompts) || !m_hasConditioningState;
    if (conditioningFrame && promptsEmpty(prompts) && !m_hasConditioningState) {
        std::cerr << "[WARN] inferMultiFrame => first tracker frame requires prompts.\n";
        return Image<float>(originalImageSize.width, originalImageSize.height, 1);
    }

    try {
        const bool denseMaskConditioning = conditioningFrame && hasMaskPrompt(prompts);
        PreparedSAM3MaskPrompt preparedMask;
        if (denseMaskConditioning
            && !prepareMaskPrompt(prompts, originalImageSize, &preparedMask)) {
            return Image<float>(originalImageSize.width, originalImageSize.height, 1);
        }

        const Ort::Value& highRes0 = encoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)];
        const Ort::Value& highRes1 = encoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)];
        const Ort::Value& currentVisionFeat = encoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)];
        std::vector<int64_t> currentVisionShape;
        std::vector<float> currentVisionValues;
        extractTensorData(currentVisionFeat, currentVisionValues, currentVisionShape);

        std::vector<Ort::Value> decoderInputs;
        double attnTimeMs = 0.0;
        bool isMaskFromPoints = conditioningFrame;
        std::vector<float> pointCoordsStorage;
        std::vector<int32_t> pointLabelsStorage;
        std::vector<float> imageEmbedStorage;
        std::vector<Ort::Value> memoryAttentionOutputs;

        if (conditioningFrame) {
            buildTrackerPromptInputs(prompts, originalImageSize, &pointCoordsStorage, &pointLabelsStorage);
            if (pointLabelsStorage.empty() && denseMaskConditioning) {
                pointCoordsStorage = preparedMask.fallbackPointCoords;
                pointLabelsStorage = preparedMask.fallbackPointLabels;
            }
            if (pointLabelsStorage.empty()) {
                std::cerr << "[WARN] inferMultiFrame => conditioning frame has no prompt tensors.\n";
                return Image<float>(originalImageSize.width, originalImageSize.height, 1);
            }

            imageEmbedStorage = buildNoMemoryImageEmbedding(currentVisionFeat);
            const std::vector<int64_t> pointCoordsShape = {1, static_cast<int64_t>(pointLabelsStorage.size()), 2};
            const std::vector<int64_t> pointLabelsShape = {1, static_cast<int64_t>(pointLabelsStorage.size())};
            decoderInputs.push_back(createTensor<float>(m_memoryInfo, pointCoordsStorage, pointCoordsShape));
            decoderInputs.push_back(createTensor<int32_t>(m_memoryInfo, pointLabelsStorage, pointLabelsShape));
            decoderInputs.push_back(createTensor<float>(m_memoryInfo, imageEmbedStorage, currentVisionShape));
        } else {
            buildMemoryInputBuffers(m_segmentFrameIndex);
            const auto attnStart = std::chrono::steady_clock::now();
            std::vector<Ort::Value> memoryInputs;
            memoryInputs.push_back(createTensorView<float>(
                m_memoryInfo,
                encoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)].GetTensorMutableData<float>(),
                currentVisionShape));
            memoryInputs.push_back(createTensor<float>(
                m_memoryInfo,
                m_constants.currentVisionPosEmbed,
                m_constants.currentVisionPosEmbedShape));
            memoryInputs.push_back(createTensor<float>(
                m_memoryInfo,
                m_memoryObjPtrsScratch,
                {static_cast<int64_t>(m_staticNumObjPtrs), 256}));
            memoryInputs.push_back(createTensor<float>(
                m_memoryInfo,
                m_memoryObjTposScratch,
                {static_cast<int64_t>(m_staticNumObjPtrs)}));
            memoryInputs.push_back(createTensor<float>(
                m_memoryInfo,
                m_memoryMaskFeatsScratch,
                {static_cast<int64_t>(m_staticNumMemFrames), 64, 72, 72}));
            memoryInputs.push_back(createTensor<float>(
                m_memoryInfo,
                m_memoryMaskPosScratch,
                {static_cast<int64_t>(m_staticNumMemFrames), 64, 72, 72}));
            memoryInputs.push_back(createTensor<int64_t>(
                m_memoryInfo,
                m_memoryMaskTposScratch,
                {static_cast<int64_t>(m_staticNumMemFrames)}));

            auto memoryAttentionResult = runSession(
                m_memoryAttentionSession.get(),
                m_memoryAttentionInputNames,
                m_memoryAttentionOutputNames,
                memoryInputs,
                "memoryAttention");
            attnTimeMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - attnStart).count();
            if (memoryAttentionResult.index() == 1) {
                std::cerr << std::get<std::string>(memoryAttentionResult) << '\n';
                return Image<float>();
            }

            memoryAttentionOutputs = std::move(std::get<0>(memoryAttentionResult));
            if (memoryAttentionOutputs.size() <= static_cast<size_t>(m_memoryAttentionFusedFeatIndex)) {
                std::cerr << "[ERROR] inferMultiFrame => memory attention returned no fused_feat output.\n";
                return Image<float>();
            }

            Ort::Value& fusedFeat = memoryAttentionOutputs[static_cast<size_t>(m_memoryAttentionFusedFeatIndex)];
            const auto fusedShape = fusedFeat.GetTensorTypeAndShapeInfo().GetShape();
            pointCoordsStorage.clear();
            pointLabelsStorage.clear();
            decoderInputs.push_back(createTensor<float>(m_memoryInfo, pointCoordsStorage, {1, 0, 2}));
            decoderInputs.push_back(createTensor<int32_t>(m_memoryInfo, pointLabelsStorage, {1, 0}));
            decoderInputs.push_back(createTensorView<float>(
                m_memoryInfo,
                fusedFeat.GetTensorMutableData<float>(),
                fusedShape));
            isMaskFromPoints = false;
        }

        decoderInputs.push_back(createTensorView<float>(
            m_memoryInfo,
            encoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)].GetTensorMutableData<float>(),
            highRes0.GetTensorTypeAndShapeInfo().GetShape()));
        decoderInputs.push_back(createTensorView<float>(
            m_memoryInfo,
            encoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)].GetTensorMutableData<float>(),
            highRes1.GetTensorTypeAndShapeInfo().GetShape()));

        const auto decoderStart = std::chrono::steady_clock::now();
        auto decoderResult = runSession(
            m_trackerDecoderSession.get(),
            m_trackerDecoderInputNames,
            m_trackerDecoderOutputNames,
            decoderInputs,
            conditioningFrame ? "trackerConditioningDecoder" : "trackerPropagationDecoder");
        const double decoderTimeMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decoderStart).count();
        if (decoderResult.index() == 1) {
            std::cerr << std::get<std::string>(decoderResult) << '\n';
            return Image<float>();
        }

        auto decoderOutputs = std::move(std::get<0>(decoderResult));
        if (decoderOutputs.size() <= static_cast<size_t>(std::max(m_trackerDecoderPredMaskHighResIndex, m_trackerDecoderObjectScoreIndex))) {
            std::cerr << "[ERROR] inferMultiFrame => tracker decoder returned insufficient outputs.\n";
            return Image<float>();
        }

        const auto memoryEncoderStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memoryEncoderInputs;
        if (denseMaskConditioning) {
            memoryEncoderInputs.push_back(createTensor<float>(
                m_memoryInfo,
                preparedMask.maskLogitsHighRes,
                preparedMask.maskLogitsShape));
        } else {
            memoryEncoderInputs.push_back(createTensorView<float>(
                m_memoryInfo,
                decoderOutputs[static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)].GetTensorMutableData<float>(),
                decoderOutputs[static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)].GetTensorTypeAndShapeInfo().GetShape()));
        }
        memoryEncoderInputs.push_back(createTensorView<float>(
            m_memoryInfo,
            encoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)].GetTensorMutableData<float>(),
            currentVisionShape));
        std::vector<float> objectPresentLogitsStorage;
        if (denseMaskConditioning) {
            objectPresentLogitsStorage = {1.0f};
            memoryEncoderInputs.push_back(createTensor<float>(
                m_memoryInfo,
                objectPresentLogitsStorage,
                {1, 1}));
        } else {
            memoryEncoderInputs.push_back(createTensorView<float>(
                m_memoryInfo,
                decoderOutputs[static_cast<size_t>(m_trackerDecoderObjectScoreIndex)].GetTensorMutableData<float>(),
                decoderOutputs[static_cast<size_t>(m_trackerDecoderObjectScoreIndex)].GetTensorTypeAndShapeInfo().GetShape()));
        }
        const std::vector<float> isMaskFromPointsTensor = {isMaskFromPoints ? 1.0f : 0.0f};
        memoryEncoderInputs.push_back(createTensor<float>(
            m_memoryInfo,
            isMaskFromPointsTensor,
            {1}));

        auto memoryEncoderResult = runSession(
            m_memoryEncoderSession.get(),
            m_memoryEncoderInputNames,
            m_memoryEncoderOutputNames,
            memoryEncoderInputs,
            conditioningFrame ? "conditioningMemoryEncoder" : "propagationMemoryEncoder");
        const double memoryEncoderTimeMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - memoryEncoderStart).count();
        if (memoryEncoderResult.index() == 1) {
            std::cerr << std::get<std::string>(memoryEncoderResult) << '\n';
            return Image<float>();
        }

        auto memoryEncoderOutputs = std::move(std::get<0>(memoryEncoderResult));
        TrackerFrameState frameState = captureTrackerState(
            decoderOutputs,
            memoryEncoderOutputs,
            m_segmentFrameIndex);
        if (conditioningFrame) {
            m_conditioningState = frameState;
            m_hasConditioningState = true;
            m_nonConditioningStates.clear();
        } else {
            appendNonConditioningState(frameState);
        }

        const Image<float> mask = denseMaskConditioning
            ? preparedMask.originalMask
            : createTrackerMaskFromHighRes(
                decoderOutputs[static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)],
                originalImageSize);
        ++m_segmentFrameIndex;

        std::cout << "[INFO] Frame times => Enc: " << encTimeMs
                  << " ms, Attn: " << attnTimeMs
                  << " ms, Dec: " << decoderTimeMs
                  << " ms, MemEnc: " << memoryEncoderTimeMs << " ms\n";
        return mask;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] inferMultiFrame => " << error.what() << '\n';
        return Image<float>();
    }
}
