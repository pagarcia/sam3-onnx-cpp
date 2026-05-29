#include "SAM3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

size_t safeMinCount(size_t a, size_t b)
{
    return std::min(a, b);
}

double frameElapsedMs(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

double trackerStateObjectProbability(const smseg_sam3::TrackerFrameState& state)
{
    return 1.0 / (1.0 + std::exp(-double(state.objectScoreLogit)));
}

double trackerStateMemoryQuality(const smseg_sam3::TrackerFrameState& state)
{
    const double objectProb = trackerStateObjectProbability(state);
    const double effIou =
        state.hasEffIouScore
            ? std::clamp((double(state.effIouScore) + 1.0) * 0.5, 0.0, 1.0)
            : objectProb;
    return 0.65 * objectProb + 0.35 * effIou;
}

std::vector<std::size_t> selectMemoryStateIndices(
    const std::deque<smseg_sam3::TrackerFrameState>& states,
    std::size_t limit,
    int currentFrameIndex)
{
    std::vector<std::size_t> selected;
    if (states.empty() || limit == 0) {
        return selected;
    }

    selected.reserve(std::min(limit, states.size()));
    if (states.size() <= limit) {
        for (std::size_t i = states.size(); i > 0; --i) {
            selected.push_back(i - 1u);
        }
        return selected;
    }

    std::vector<std::uint8_t> used(states.size(), std::uint8_t(0));
    const std::size_t recentKeep =
        std::min<std::size_t>(std::min<std::size_t>(2u, limit), states.size());
    for (std::size_t k = 0; k < recentKeep; ++k) {
        const std::size_t idx = states.size() - 1u - k;
        selected.push_back(idx);
        used[idx] = std::uint8_t(1);
    }

    while (selected.size() < limit && selected.size() < states.size()) {
        double bestScore = -1.0e30;
        std::size_t bestIndex = states.size();
        for (std::size_t i = 0; i < states.size(); ++i) {
            if (used[i]) {
                continue;
            }

            const smseg_sam3::TrackerFrameState& state = states[i];
            int minFrameDistance = std::numeric_limits<int>::max();
            for (const std::size_t selectedIndex : selected) {
                minFrameDistance =
                    std::min(minFrameDistance,
                             std::abs(state.frameIndex - states[selectedIndex].frameIndex));
            }
            if (minFrameDistance == std::numeric_limits<int>::max()) {
                minFrameDistance = 0;
            }

            const int age = std::max(0, currentFrameIndex - state.frameIndex);
            const double quality = trackerStateMemoryQuality(state);
            const double diversityBonus =
                0.020 * double(std::min(minFrameDistance, 12));
            const double longMemoryBonus =
                0.006 * double(std::min(age, 24));
            const double score = quality + diversityBonus + longMemoryBonus;
            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }

        if (bestIndex >= states.size()) {
            break;
        }
        selected.push_back(bestIndex);
        used[bestIndex] = std::uint8_t(1);
    }

    return selected;
}

Image<float> resizeAndThresholdMask(const float* maskData,
                                    int maskWidth,
                                    int maskHeight,
                                    int targetWidth,
                                    int targetHeight,
                                    float threshold)
{
    if (!maskData || maskWidth <= 0 || maskHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
        return Image<float>();
    }

    Image<float> lowRes(maskWidth, maskHeight, 1);
    std::copy(maskData, maskData + static_cast<std::size_t>(maskWidth) * maskHeight, lowRes.getData().begin());
    Image<float> resized = lowRes.resize(targetWidth, targetHeight);
    for (float& value : resized.getData()) {
        value = value > threshold ? 1.0f : 0.0f;
    }
    return resized;
}

std::vector<Image<float>> resizeAndThresholdMaskPlanes(const Ort::Value& tensor,
                                                       int targetWidth,
                                                       int targetHeight,
                                                       float threshold)
{
    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() < 3 || targetWidth <= 0 || targetHeight <= 0) {
        return {};
    }

    const int64_t maskHeight64 = shape[shape.size() - 2];
    const int64_t maskWidth64 = shape[shape.size() - 1];
    if (maskWidth64 <= 0 || maskHeight64 <= 0) {
        return {};
    }

    std::size_t planes = 1;
    for (std::size_t i = 0; i + 2 < shape.size(); ++i) {
        if (shape[i] <= 0) {
            return {};
        }
        planes *= static_cast<std::size_t>(shape[i]);
    }

    const int maskWidth = static_cast<int>(maskWidth64);
    const int maskHeight = static_cast<int>(maskHeight64);
    const std::size_t planeSize =
        static_cast<std::size_t>(maskWidth) * static_cast<std::size_t>(maskHeight);
    const float* maskData = tensor.GetTensorData<float>();
    if (!maskData || planeSize == 0 || planes == 0) {
        return {};
    }

    std::vector<Image<float>> masks;
    masks.reserve(planes);
    for (std::size_t i = 0; i < planes; ++i) {
        masks.push_back(resizeAndThresholdMask(
            maskData + i * planeSize,
            maskWidth,
            maskHeight,
            targetWidth,
            targetHeight,
            threshold));
    }
    return masks;
}

std::vector<float> tensorFloatValues(const Ort::Value& tensor)
{
    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    std::size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            return {};
        }
        count *= static_cast<std::size_t>(dim);
    }

    const float* values = tensor.GetTensorData<float>();
    if (!values || count == 0) {
        return {};
    }
    return std::vector<float>(values, values + count);
}

int bestScoreIndex(const std::vector<float>& scores, std::size_t maskCount)
{
    const std::size_t count = std::min(scores.size(), maskCount);
    if (count == 0) {
        return -1;
    }

    int best = 0;
    for (std::size_t i = 1; i < count; ++i) {
        if (scores[i] > scores[static_cast<std::size_t>(best)]) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

smseg_sam3::SAM3MaskCandidates singleAuthoritativeMaskCandidate(const Image<float>& mask)
{
    smseg_sam3::SAM3MaskCandidates result;
    result.selectedMask = mask;
    if (mask.getWidth() > 0 && mask.getHeight() > 0 && mask.getChannels() == 1) {
        result.masks.push_back(mask);
        result.scores.push_back(1.0f);
        result.selectedIndex = 0;
    }
    return result;
}

bool copyMaskPlaneTensor(const Ort::Value& tensor,
                         int planeIndex,
                         std::vector<float>* logitsOut,
                         std::vector<int64_t>* shapeOut)
{
    if (!logitsOut || !shapeOut || planeIndex < 0) {
        return false;
    }
    logitsOut->clear();
    shapeOut->clear();

    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() < 3) {
        return false;
    }

    const int64_t maskHeight64 = shape[shape.size() - 2];
    const int64_t maskWidth64 = shape[shape.size() - 1];
    if (maskWidth64 <= 0 || maskHeight64 <= 0) {
        return false;
    }

    std::size_t planes = 1;
    for (std::size_t i = 0; i + 2 < shape.size(); ++i) {
        if (shape[i] <= 0) {
            return false;
        }
        planes *= static_cast<std::size_t>(shape[i]);
    }
    if (static_cast<std::size_t>(planeIndex) >= planes) {
        return false;
    }

    const std::size_t planeSize =
        static_cast<std::size_t>(maskWidth64) * static_cast<std::size_t>(maskHeight64);
    const float* tensorData = tensor.GetTensorData<float>();
    if (!tensorData || planeSize == 0) {
        return false;
    }

    const float* plane = tensorData + static_cast<std::size_t>(planeIndex) * planeSize;
    logitsOut->assign(plane, plane + planeSize);
    *shapeOut = {
        1,
        1,
        maskHeight64,
        maskWidth64,
    };
    return true;
}

bool maskToBinaryHighResLogits(const Image<float>& mask,
                               const smseg_sam3::SAM3Size& inputSize,
                               std::vector<float>* logitsOut,
                               std::vector<int64_t>* shapeOut)
{
    if (!logitsOut || !shapeOut
        || mask.getWidth() <= 0 || mask.getHeight() <= 0
        || inputSize.width <= 0 || inputSize.height <= 0) {
        return false;
    }

    Image<float> resized =
        (mask.getWidth() == inputSize.width && mask.getHeight() == inputSize.height)
            ? mask
            : mask.resize(inputSize.width, inputSize.height);
    logitsOut->assign(
        static_cast<std::size_t>(inputSize.width) * static_cast<std::size_t>(inputSize.height),
        -20.0f);
    const auto& data = resized.getData();
    const std::size_t n = std::min(logitsOut->size(), data.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] > 0.5f) {
            (*logitsOut)[i] = 20.0f;
        }
    }
    *shapeOut = {
        1,
        1,
        static_cast<int64_t>(inputSize.height),
        static_cast<int64_t>(inputSize.width),
    };
    return true;
}

Image<float> normalizeBinaryMaskToSize(const Image<float>& mask,
                                       const smseg_sam3::SAM3Size& size)
{
    if (mask.getWidth() <= 0 || mask.getHeight() <= 0
        || size.width <= 0 || size.height <= 0) {
        return Image<float>();
    }
    Image<float> out =
        (mask.getWidth() == size.width && mask.getHeight() == size.height)
            ? mask
            : mask.resize(size.width, size.height);
    for (float& value : out.getData()) {
        value = value > 0.5f ? 1.0f : 0.0f;
    }
    return out;
}

} // namespace

namespace smseg_sam3 {

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
    return resizeAndThresholdMask(
        maskData,
        static_cast<int>(shape[3]),
        static_cast<int>(shape[2]),
        originalImageSize.width,
        originalImageSize.height,
        0.0f);
}

SAM3MaskCandidates SAM3::collectTrackerMaskCandidates(
    const std::vector<Ort::Value>& decoderOutputs,
    const SAM3Size& originalImageSize) const
{
    SAM3MaskCandidates result;
    if (m_trackerDecoderPredMaskHighResIndex < 0
        || decoderOutputs.size() <= static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)) {
        return result;
    }

    result.selectedMask = createTrackerMaskFromHighRes(
        decoderOutputs[static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)],
        originalImageSize);

    if (m_trackerDecoderPredMultimasksHighResIndex >= 0
        && decoderOutputs.size() > static_cast<size_t>(m_trackerDecoderPredMultimasksHighResIndex)) {
        result.masks = resizeAndThresholdMaskPlanes(
            decoderOutputs[static_cast<size_t>(m_trackerDecoderPredMultimasksHighResIndex)],
            originalImageSize.width,
            originalImageSize.height,
            0.0f);
    }
    if (m_trackerDecoderIouScoresIndex >= 0
        && decoderOutputs.size() > static_cast<size_t>(m_trackerDecoderIouScoresIndex)) {
        result.scores = tensorFloatValues(
            decoderOutputs[static_cast<size_t>(m_trackerDecoderIouScoresIndex)]);
    }
    result.selectedIndex = bestScoreIndex(result.scores, result.masks.size());
    return result;
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

    const std::size_t nonConditioningMemorySlots =
        memoryRow < memorySlotCount ? memorySlotCount - memoryRow : 0u;
    const std::vector<std::size_t> memoryIndices =
        selectMemoryStateIndices(m_nonConditioningStates,
                                 nonConditioningMemorySlots,
                                 frameIndex);
    int relativeMemoryIndex = 0;
    for (const std::size_t stateIndex : memoryIndices) {
        if (stateIndex >= m_nonConditioningStates.size() || memoryRow >= memorySlotCount) {
            continue;
        }
        const TrackerFrameState& state = m_nonConditioningStates[stateIndex];
        const size_t copyCount = safeMinCount(featurePlaneSize, state.maskmemFeatures.size());
        const size_t posCopyCount = safeMinCount(featurePlaneSize, state.maskmemPosEnc.size());
        std::copy_n(
            state.maskmemFeatures.begin(),
            copyCount,
            m_memoryMaskFeatsScratch.begin() + static_cast<ptrdiff_t>(memoryRow * featurePlaneSize));
        std::copy_n(
            state.maskmemPosEnc.begin(),
            posCopyCount,
            m_memoryMaskPosScratch.begin() + static_cast<ptrdiff_t>(memoryRow * featurePlaneSize));
        const int temporalAge =
            std::max(0, frameIndex - state.frameIndex - 1);
        m_memoryMaskTposScratch[memoryRow] =
            std::clamp(temporalAge,
                       relativeMemoryIndex,
                       std::max(relativeMemoryIndex,
                                std::max(0, m_constants.numMaskmem - 2)));
        ++memoryRow;
        ++relativeMemoryIndex;
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

    const std::size_t nonConditioningObjSlots =
        objRow < objSlotCount ? objSlotCount - objRow : 0u;
    const std::vector<std::size_t> objIndices =
        selectMemoryStateIndices(m_nonConditioningStates,
                                 nonConditioningObjSlots,
                                 frameIndex);
    int relativePointerIndex = 1;
    for (const std::size_t stateIndex : objIndices) {
        if (stateIndex >= m_nonConditioningStates.size() || objRow >= objSlotCount) {
            continue;
        }
        const TrackerFrameState& state = m_nonConditioningStates[stateIndex];
        const size_t copyCount = safeMinCount(objPtrSize, state.objPtr.size());
        std::copy_n(
            state.objPtr.begin(),
            copyCount,
            m_memoryObjPtrsScratch.begin() + static_cast<ptrdiff_t>(objRow * objPtrSize));
        const int temporalAge =
            std::max(relativePointerIndex, frameIndex - state.frameIndex);
        m_memoryObjTposScratch[objRow] = static_cast<float>(temporalAge);
        ++objRow;
        ++relativePointerIndex;
    }
}

Image<float> SAM3::previewConditioningFrame(const SAM3Size& originalImageSize,
                                            const SAM3Prompts& prompts)
{
    return previewConditioningFrameCandidates(originalImageSize, prompts).selectedMask;
}

SAM3MaskCandidates SAM3::previewConditioningFrameCandidates(const SAM3Size& originalImageSize,
                                                            const SAM3Prompts& prompts)
{
    SAM3MaskCandidates result;

    if (!m_trackerDecoderSession || !m_hasVideoConstants) {
        std::cerr << "[ERROR] previewConditioningFrame => tracker sessions are not initialized.\n";
        return result;
    }

    const int requiredMaxIndex = std::max({m_encoderImageEmb0Index, m_encoderImageEmb1Index, m_encoderImageEmb2Index});
    if (requiredMaxIndex < 0 || m_cachedEncoderOutputs.size() <= static_cast<size_t>(requiredMaxIndex)) {
        std::cerr << "[ERROR] previewConditioningFrame => encoder outputs are not cached.\n";
        return result;
    }

    if (promptsEmpty(prompts)) {
        result.selectedMask = Image<float>(originalImageSize.width, originalImageSize.height, 1);
        return result;
    }

    if (hasMaskPrompt(prompts)) {
        PreparedSAM3MaskPrompt preparedMask;
        if (prepareMaskPrompt(prompts, originalImageSize, &preparedMask)) {
            result = singleAuthoritativeMaskCandidate(preparedMask.originalMask);
            return result;
        }
        result.selectedMask = Image<float>(originalImageSize.width, originalImageSize.height, 1);
        return result;
    }

    try {
        std::vector<float> pointCoords;
        std::vector<int32_t> pointLabels;
        buildTrackerPromptInputs(prompts, originalImageSize, &pointCoords, &pointLabels);
        if (pointLabels.empty()) {
            result.selectedMask = Image<float>(originalImageSize.width, originalImageSize.height, 1);
            return result;
        }

        const std::vector<int64_t> pointCoordsShape = {1, static_cast<int64_t>(pointLabels.size()), 2};
        const std::vector<int64_t> pointLabelsShape = {1, static_cast<int64_t>(pointLabels.size())};
        const std::vector<int64_t> currentVisionShape =
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)]
                .GetTensorTypeAndShapeInfo()
                .GetShape();
        const std::vector<float>& imageEmbed = buildNoMemoryImageEmbedding(
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)],
            currentVisionShape);

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

        auto sessionResult = runSession(
            m_trackerDecoderSession.get(),
            m_trackerDecoderInputNames,
            m_trackerDecoderOutputNames,
            inputs,
            "trackerPreview");
        if (sessionResult.index() == 1) {
            std::cerr << std::get<std::string>(sessionResult) << '\n';
            return SAM3MaskCandidates{};
        }

        auto outputs = std::move(std::get<0>(sessionResult));
        if (outputs.size() <= static_cast<size_t>(m_trackerDecoderPredMaskHighResIndex)) {
            std::cerr << "[ERROR] previewConditioningFrame => decoder did not return pred_mask_high_res.\n";
            return SAM3MaskCandidates{};
        }
        result = collectTrackerMaskCandidates(outputs, originalImageSize);
        return result;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] previewConditioningFrame => " << error.what() << '\n';
        return SAM3MaskCandidates{};
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

bool SAM3::lastTrackerFrameState(TrackerFrameState* stateOut) const
{
    if (!stateOut || !m_hasLastTrackerFrameState) {
        return false;
    }
    *stateOut = m_lastTrackerFrameState;
    return true;
}

void SAM3::setTrackerMaskSelectionCallback(TrackerMaskSelectionCallback callback)
{
    m_trackerMaskSelectionCallback = std::move(callback);
}

bool SAM3::lastTrackerMaskCandidates(SAM3MaskCandidates* candidatesOut) const
{
    if (!candidatesOut || !m_hasLastTrackerMaskCandidates) {
        return false;
    }
    *candidatesOut = m_lastTrackerMaskCandidates;
    return true;
}

Image<float> SAM3::inferMultiFrameWithEncoderOutputs(std::vector<Ort::Value>& encoderOutputs,
                                                     const SAM3Size& originalImageSize,
                                                     const SAM3Prompts& prompts,
                                                     double encTimeMs)
{
    const auto totalStart = std::chrono::steady_clock::now();
    SAM3FrameTimings timings;
    timings.frameIndex = m_segmentFrameIndex;
    timings.encMs = encTimeMs;

    m_hasLastTrackerFrameState = false;
    m_lastTrackerMaskCandidates = SAM3MaskCandidates();
    m_hasLastTrackerMaskCandidates = false;
    m_lastFrameTimings = SAM3FrameTimings();
    m_hasLastFrameTimings = false;

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
    timings.conditioningFrame = conditioningFrame;
    if (conditioningFrame && promptsEmpty(prompts) && !m_hasConditioningState) {
        std::cerr << "[WARN] inferMultiFrame => first tracker frame requires prompts.\n";
        return Image<float>(originalImageSize.width, originalImageSize.height, 1);
    }

    try {
        const bool denseMaskConditioning = conditioningFrame && hasMaskPrompt(prompts);
        timings.denseMaskConditioning = denseMaskConditioning;
        PreparedSAM3MaskPrompt preparedMask;
        if (denseMaskConditioning) {
            const auto maskPrepStart = std::chrono::steady_clock::now();
            const bool maskPrepared =
                prepareMaskPrompt(prompts, originalImageSize, &preparedMask);
            timings.maskPrepMs += frameElapsedMs(maskPrepStart);
            if (!maskPrepared) {
                return Image<float>(originalImageSize.width, originalImageSize.height, 1);
            }
        }

        const Ort::Value& highRes0 = encoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)];
        const Ort::Value& highRes1 = encoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)];
        const Ort::Value& currentVisionFeat = encoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)];
        const std::vector<int64_t> currentVisionShape =
            currentVisionFeat.GetTensorTypeAndShapeInfo().GetShape();

        std::vector<Ort::Value> decoderInputs;
        double attnTimeMs = 0.0;
        bool isMaskFromPoints = conditioningFrame;
        std::vector<float> pointCoordsStorage;
        std::vector<int32_t> pointLabelsStorage;
        const std::vector<float>* imageEmbedStorage = nullptr;
        std::vector<Ort::Value> memoryAttentionOutputs;

        if (conditioningFrame) {
            const auto promptStart = std::chrono::steady_clock::now();
            buildTrackerPromptInputs(prompts, originalImageSize, &pointCoordsStorage, &pointLabelsStorage);
            if (pointLabelsStorage.empty() && denseMaskConditioning) {
                pointCoordsStorage = preparedMask.fallbackPointCoords;
                pointLabelsStorage = preparedMask.fallbackPointLabels;
            }
            timings.promptBuildMs += frameElapsedMs(promptStart);
            if (pointLabelsStorage.empty()) {
                std::cerr << "[WARN] inferMultiFrame => conditioning frame has no prompt tensors.\n";
                return Image<float>(originalImageSize.width, originalImageSize.height, 1);
            }

            const auto embedStart = std::chrono::steady_clock::now();
            imageEmbedStorage =
                &buildNoMemoryImageEmbedding(currentVisionFeat, currentVisionShape);
            timings.noMemoryEmbedMs += frameElapsedMs(embedStart);
            const std::vector<int64_t> pointCoordsShape = {1, static_cast<int64_t>(pointLabelsStorage.size()), 2};
            const std::vector<int64_t> pointLabelsShape = {1, static_cast<int64_t>(pointLabelsStorage.size())};
            decoderInputs.push_back(createTensor<float>(m_memoryInfo, pointCoordsStorage, pointCoordsShape));
            decoderInputs.push_back(createTensor<int32_t>(m_memoryInfo, pointLabelsStorage, pointLabelsShape));
            decoderInputs.push_back(createTensor<float>(m_memoryInfo, *imageEmbedStorage, currentVisionShape));
        } else {
            const auto memoryBuildStart = std::chrono::steady_clock::now();
            buildMemoryInputBuffers(m_segmentFrameIndex);
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
            timings.memoryBuildMs += frameElapsedMs(memoryBuildStart);

            const auto attnStart = std::chrono::steady_clock::now();
            auto memoryAttentionResult = runSession(
                m_memoryAttentionSession.get(),
                m_memoryAttentionInputNames,
                m_memoryAttentionOutputNames,
                memoryInputs,
                "memoryAttention");
            attnTimeMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - attnStart).count();
            timings.attnMs = attnTimeMs;
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
        timings.decoderMs = decoderTimeMs;
        if (decoderResult.index() == 1) {
            std::cerr << std::get<std::string>(decoderResult) << '\n';
            return Image<float>();
        }

        auto decoderOutputs = std::move(std::get<0>(decoderResult));
        if (decoderOutputs.size() <= static_cast<size_t>(std::max(m_trackerDecoderPredMaskHighResIndex, m_trackerDecoderObjectScoreIndex))) {
            std::cerr << "[ERROR] inferMultiFrame => tracker decoder returned insufficient outputs.\n";
            return Image<float>();
        }
        const auto candidateStart = std::chrono::steady_clock::now();
        SAM3MaskCandidates trackerMaskCandidates =
            denseMaskConditioning
                ? singleAuthoritativeMaskCandidate(preparedMask.originalMask)
                : collectTrackerMaskCandidates(decoderOutputs, originalImageSize);
        timings.candidateMs += frameElapsedMs(candidateStart);
        timings.candidateCount =
            static_cast<int>(trackerMaskCandidates.masks.size());
        Image<float> selectedTrackerMask = trackerMaskCandidates.selectedMask;
        int selectedTrackerMaskIndex = trackerMaskCandidates.selectedIndex;
        std::vector<float> selectedTrackerMaskLogitsHighRes;
        std::vector<int64_t> selectedTrackerMaskLogitsShape;
        if (!denseMaskConditioning
            && trackerMaskCandidates.hasCandidates()
            && m_trackerMaskSelectionCallback) {
            timings.usedSelectionCallback = true;
            const auto selectionStart = std::chrono::steady_clock::now();
            const SAM3MaskSelection selection =
                m_trackerMaskSelectionCallback(trackerMaskCandidates);
            timings.selectionMs += frameElapsedMs(selectionStart);
            if (selection.hasOverrideMask()) {
                selectedTrackerMask =
                    normalizeBinaryMaskToSize(selection.overrideMask, originalImageSize);
                selectedTrackerMaskIndex = selection.candidateIndex;
                const auto logitsStart = std::chrono::steady_clock::now();
                (void)maskToBinaryHighResLogits(
                    selectedTrackerMask,
                    getInputSize(),
                    &selectedTrackerMaskLogitsHighRes,
                    &selectedTrackerMaskLogitsShape);
                timings.selectionLogitsMs += frameElapsedMs(logitsStart);
            } else if (selection.candidateIndex >= 0
                       && static_cast<std::size_t>(selection.candidateIndex)
                              < trackerMaskCandidates.masks.size()) {
                selectedTrackerMaskIndex = selection.candidateIndex;
                selectedTrackerMask =
                    trackerMaskCandidates.masks[static_cast<std::size_t>(selection.candidateIndex)];
                const auto logitsStart = std::chrono::steady_clock::now();
                if (m_trackerDecoderPredMultimasksHighResIndex < 0
                    || decoderOutputs.size()
                           <= static_cast<std::size_t>(m_trackerDecoderPredMultimasksHighResIndex)
                    || !copyMaskPlaneTensor(
                           decoderOutputs[static_cast<std::size_t>(
                               m_trackerDecoderPredMultimasksHighResIndex)],
                           selection.candidateIndex,
                           &selectedTrackerMaskLogitsHighRes,
                           &selectedTrackerMaskLogitsShape)) {
                    (void)maskToBinaryHighResLogits(
                        selectedTrackerMask,
                        getInputSize(),
                        &selectedTrackerMaskLogitsHighRes,
                        &selectedTrackerMaskLogitsShape);
                }
                timings.selectionLogitsMs += frameElapsedMs(logitsStart);
            }
        }

        const auto memoryEncoderStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memoryEncoderInputs;
        if (denseMaskConditioning) {
            memoryEncoderInputs.push_back(createTensor<float>(
                m_memoryInfo,
                preparedMask.maskLogitsHighRes,
                preparedMask.maskLogitsShape));
        } else if (!selectedTrackerMaskLogitsHighRes.empty()
                   && !selectedTrackerMaskLogitsShape.empty()) {
            memoryEncoderInputs.push_back(createTensor<float>(
                m_memoryInfo,
                selectedTrackerMaskLogitsHighRes,
                selectedTrackerMaskLogitsShape));
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
            objectPresentLogitsStorage = {10.0f};
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
        timings.memMs = memoryEncoderTimeMs;
        if (memoryEncoderResult.index() == 1) {
            std::cerr << std::get<std::string>(memoryEncoderResult) << '\n';
            return Image<float>();
        }

        auto memoryEncoderOutputs = std::move(std::get<0>(memoryEncoderResult));
        const auto captureStateStart = std::chrono::steady_clock::now();
        TrackerFrameState frameState = captureTrackerState(
            decoderOutputs,
            memoryEncoderOutputs,
            m_segmentFrameIndex);
        if (denseMaskConditioning) {
            frameState.objectScoreLogit = 10.0f;
            frameState.effIouScore = 1.0f;
            frameState.hasEffIouScore = true;
        }
        timings.captureStateMs += frameElapsedMs(captureStateStart);
        const auto stateUpdateStart = std::chrono::steady_clock::now();
        m_lastTrackerFrameState = frameState;
        m_hasLastTrackerFrameState = true;
        if (conditioningFrame) {
            m_conditioningState = frameState;
            m_hasConditioningState = true;
            m_nonConditioningStates.clear();
        } else {
            appendNonConditioningState(frameState);
        }

        Image<float> mask = denseMaskConditioning
            ? preparedMask.originalMask
            : selectedTrackerMask;
        trackerMaskCandidates.selectedIndex = selectedTrackerMaskIndex;
        trackerMaskCandidates.selectedMask = mask;
        m_lastTrackerMaskCandidates = std::move(trackerMaskCandidates);
        m_hasLastTrackerMaskCandidates = m_lastTrackerMaskCandidates.hasCandidates();
        ++m_segmentFrameIndex;
        timings.stateUpdateMs += frameElapsedMs(stateUpdateStart);
        timings.totalMs = encTimeMs + frameElapsedMs(totalStart);
        m_lastFrameTimings = timings;
        m_hasLastFrameTimings = true;
        return mask;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] inferMultiFrame => " << error.what() << '\n';
        return Image<float>();
    }
}

} // namespace smseg_sam3
