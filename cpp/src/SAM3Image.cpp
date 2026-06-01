#include "SAM3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

Image<float> thresholdImage(Image<float> image, float threshold)
{
    for (float& value : image.getData()) {
        value = value > threshold ? 1.0f : 0.0f;
    }
    return image;
}

Image<float> resizeAndThreshold(const Image<float>& image,
                                int width,
                                int height,
                                float threshold)
{
    if (image.getWidth() <= 0 || image.getHeight() <= 0 || width <= 0 || height <= 0) {
        return Image<float>();
    }
    const Image<float> resized =
        image.getWidth() == width && image.getHeight() == height
            ? image
            : image.resize(width, height);
    return thresholdImage(resized, threshold);
}

smseg_sam3::SAM3Size promptMaskInputSize(const smseg_sam3::SAM3Size& inputSize)
{
    constexpr int kSam3BackboneStride = 14;
    constexpr int kSam3MaskPromptScale = 4;
    if (inputSize.width <= 0 || inputSize.height <= 0) {
        return smseg_sam3::SAM3Size();
    }
    return smseg_sam3::SAM3Size(
        std::max(1, (inputSize.width / kSam3BackboneStride) * kSam3MaskPromptScale),
        std::max(1, (inputSize.height / kSam3BackboneStride) * kSam3MaskPromptScale));
}

} // namespace

namespace smseg_sam3 {

bool SAM3::promptsEmpty(const SAM3Prompts& prompts) const
{
    return prompts.points.empty() && prompts.rects.empty() && !hasMaskPrompt(prompts);
}

bool SAM3::hasMaskPrompt(const SAM3Prompts& prompts) const
{
    return prompts.mask.getWidth() > 0
        && prompts.mask.getHeight() > 0
        && prompts.mask.getChannels() == 1;
}

bool SAM3::prepareMaskPrompt(const SAM3Prompts& prompts,
                             const SAM3Size& originalImageSize,
                             PreparedSAM3MaskPrompt* preparedOut) const
{
    if (!preparedOut) {
        return false;
    }
    *preparedOut = PreparedSAM3MaskPrompt();

    const SAM3Size inputSize = getInputSize();
    if (!hasMaskPrompt(prompts)
        || inputSize.width <= 0 || inputSize.height <= 0
        || originalImageSize.width <= 0 || originalImageSize.height <= 0) {
        return false;
    }

    Image<float> targetMask;
    if (prompts.mask.getWidth() == originalImageSize.width
        && prompts.mask.getHeight() == originalImageSize.height) {
        preparedOut->originalMask = thresholdImage(prompts.mask, 0.5f);
        targetMask = resizeAndThreshold(preparedOut->originalMask, inputSize.width, inputSize.height, 0.5f);
    } else if (prompts.mask.getWidth() == inputSize.width
               && prompts.mask.getHeight() == inputSize.height) {
        targetMask = thresholdImage(prompts.mask, 0.5f);
    } else {
        std::cerr
            << "[ERROR] Mask prompt must match either the original image size ("
            << originalImageSize.width << "x" << originalImageSize.height
            << ") or the Engine 3 input size (" << inputSize.width << "x" << inputSize.height
            << "); got " << prompts.mask.getWidth() << "x" << prompts.mask.getHeight() << ".\n";
        return false;
    }

    int xMin = inputSize.width;
    int yMin = inputSize.height;
    int xMax = -1;
    int yMax = -1;
    preparedOut->maskLogitsHighRes.assign(
        static_cast<std::size_t>(inputSize.width) * static_cast<std::size_t>(inputSize.height),
        -20.0f);

    const auto& targetData = targetMask.getData();
    for (int y = 0; y < inputSize.height; ++y) {
        for (int x = 0; x < inputSize.width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * inputSize.width + x;
            if (index >= targetData.size() || targetData[index] <= 0.5f) {
                continue;
            }
            preparedOut->maskLogitsHighRes[index] = 20.0f;
            xMin = std::min(xMin, x);
            yMin = std::min(yMin, y);
            xMax = std::max(xMax, x);
            yMax = std::max(yMax, y);
        }
    }

    if (xMax < 0 || yMax < 0) {
        std::cerr << "[ERROR] Mask prompt cannot be empty.\n";
        return false;
    }

    preparedOut->maskLogitsShape = {
        1,
        1,
        static_cast<int64_t>(inputSize.height),
        static_cast<int64_t>(inputSize.width),
    };
    const SAM3Size promptSize = promptMaskInputSize(inputSize);
    if (promptSize.width <= 0 || promptSize.height <= 0) {
        return false;
    }
    const Image<float> promptMask =
        resizeAndThreshold(targetMask, promptSize.width, promptSize.height, 0.5f);
    preparedOut->maskPrompt = promptMask.getData();
    preparedOut->maskPromptShape = {
        1,
        1,
        static_cast<int64_t>(promptSize.height),
        static_cast<int64_t>(promptSize.width),
    };
    if (preparedOut->originalMask.getWidth() <= 0 || preparedOut->originalMask.getHeight() <= 0) {
        Image<float> logitsImage(inputSize.width, inputSize.height, 1, preparedOut->maskLogitsHighRes);
        preparedOut->originalMask =
            resizeAndThreshold(logitsImage, originalImageSize.width, originalImageSize.height, 0.0f);
    }

    const bool usePointPrompt =
        prompts.maskPromptStrategy == SAM3MaskPromptStrategy::Point
        || xMin == xMax
        || yMin == yMax;
    if (usePointPrompt) {
        preparedOut->fallbackPointCoords.push_back((xMin + xMax) * 0.5f);
        preparedOut->fallbackPointCoords.push_back((yMin + yMax) * 0.5f);
        preparedOut->fallbackPointLabels.push_back(1);
    } else {
        preparedOut->fallbackPointCoords.push_back(static_cast<float>(xMin));
        preparedOut->fallbackPointCoords.push_back(static_cast<float>(yMin));
        preparedOut->fallbackPointLabels.push_back(2);
        preparedOut->fallbackPointCoords.push_back(static_cast<float>(xMax));
        preparedOut->fallbackPointCoords.push_back(static_cast<float>(yMax));
        preparedOut->fallbackPointLabels.push_back(3);
    }
    return true;
}

void SAM3::buildImagePromptInputs(const SAM3Prompts& prompts,
                                  const SAM3Size& originalImageSize,
                                  std::vector<float>* pointsOut,
                                  std::vector<int64_t>* labelsOut,
                                  std::vector<float>* boxesOut) const
{
    pointsOut->clear();
    labelsOut->clear();
    boxesOut->clear();

    const SAM3Size inputSize = getInputSize();
    if (inputSize.width <= 0 || inputSize.height <= 0
        || originalImageSize.width <= 0 || originalImageSize.height <= 0) {
        return;
    }

    if (!prompts.rects.empty()) {
        const SAM3Rect rawRect = prompts.rects.front();
        const int x1 = std::min(rawRect.x, rawRect.x + rawRect.width);
        const int y1 = std::min(rawRect.y, rawRect.y + rawRect.height);
        const int x2 = std::max(rawRect.x, rawRect.x + rawRect.width);
        const int y2 = std::max(rawRect.y, rawRect.y + rawRect.height);
        boxesOut->push_back(x1 * static_cast<float>(inputSize.width) / originalImageSize.width);
        boxesOut->push_back(y1 * static_cast<float>(inputSize.height) / originalImageSize.height);
        boxesOut->push_back(x2 * static_cast<float>(inputSize.width) / originalImageSize.width);
        boxesOut->push_back(y2 * static_cast<float>(inputSize.height) / originalImageSize.height);
        return;
    }

    const size_t pointCount = std::min(prompts.points.size(), prompts.pointLabels.size());
    for (size_t index = 0; index < pointCount; ++index) {
        pointsOut->push_back(
            prompts.points[index].x * static_cast<float>(inputSize.width) / originalImageSize.width);
        pointsOut->push_back(
            prompts.points[index].y * static_cast<float>(inputSize.height) / originalImageSize.height);
        labelsOut->push_back(static_cast<int64_t>(prompts.pointLabels[index]));
    }

    if (labelsOut->empty() && hasMaskPrompt(prompts)) {
        PreparedSAM3MaskPrompt preparedMask;
        if (!prepareMaskPrompt(prompts, originalImageSize, &preparedMask)) {
            return;
        }

        if (preparedMask.fallbackPointLabels.size() == 2
            && preparedMask.fallbackPointLabels[0] == 2
            && preparedMask.fallbackPointLabels[1] == 3) {
            *boxesOut = preparedMask.fallbackPointCoords;
            return;
        }

        *pointsOut = preparedMask.fallbackPointCoords;
        labelsOut->assign(
            preparedMask.fallbackPointLabels.begin(),
            preparedMask.fallbackPointLabels.end());
    }
}

Image<float> SAM3::createImageMaskFromLogits(const float* logits,
                                             int maskWidth,
                                             int maskHeight,
                                             const SAM3Size& originalImageSize) const
{
    if (!logits || maskWidth <= 0 || maskHeight <= 0) {
        return Image<float>();
    }

    const SAM3Size inputSize = getInputSize();
    Image<float> lowRes(maskWidth, maskHeight, 1);
    std::copy(logits, logits + static_cast<std::size_t>(maskWidth) * maskHeight, lowRes.getData().begin());
    Image<float> inputRes = lowRes.resize(inputSize.width, inputSize.height);
    Image<float> originalRes = inputRes.resize(originalImageSize.width, originalImageSize.height);
    return thresholdImage(originalRes, 0.0f);
}

Image<float> SAM3::createImageMaskFromDecoderOutput(const Ort::Value& predMasks,
                                                    const Ort::Value& iouScores,
                                                    const SAM3Size& originalImageSize) const
{
    const auto maskShape = predMasks.GetTensorTypeAndShapeInfo().GetShape();
    const auto scoreShape = iouScores.GetTensorTypeAndShapeInfo().GetShape();
    if (maskShape.size() != 4 && maskShape.size() != 5) {
        std::cerr << "[ERROR] Unexpected image decoder mask output shape.\n";
        return Image<float>();
    }

    const int64_t numMasks = maskShape.size() == 5 ? maskShape[2] : maskShape[1];
    const int64_t maskHeight = maskShape.size() == 5 ? maskShape[3] : maskShape[2];
    const int64_t maskWidth = maskShape.size() == 5 ? maskShape[4] : maskShape[3];
    const float* maskData = predMasks.GetTensorData<float>();
    const float* scoreData = iouScores.GetTensorData<float>();
    if (!maskData || !scoreData || numMasks <= 0 || maskHeight <= 0 || maskWidth <= 0) {
        std::cerr << "[ERROR] Decoder returned empty mask data.\n";
        return Image<float>();
    }

    const size_t scoreCount = computeElementCount(scoreShape);
    int bestMaskIndex = 0;
    if (scoreCount >= static_cast<size_t>(numMasks)) {
        float bestScore = scoreData[0];
        for (int64_t index = 1; index < numMasks; ++index) {
            if (scoreData[index] > bestScore) {
                bestScore = scoreData[index];
                bestMaskIndex = static_cast<int>(index);
            }
        }
    }

    const size_t maskPlaneSize = static_cast<size_t>(maskHeight * maskWidth);
    return createImageMaskFromLogits(
        maskData + static_cast<size_t>(bestMaskIndex) * maskPlaneSize,
        static_cast<int>(maskWidth),
        static_cast<int>(maskHeight),
        originalImageSize);
}

Image<float> SAM3::inferSingleFrame(const SAM3Size& originalImageSize,
                                    const SAM3Prompts& prompts)
{
    if (!m_imageDecoderSession) {
        std::cerr << "[ERROR] inferSingleFrame => image decoder session is not initialized.\n";
        return Image<float>();
    }

    const int requiredMaxIndex = std::max({m_encoderImageEmb0Index, m_encoderImageEmb1Index, m_encoderImageEmb2Index});
    if (requiredMaxIndex < 0 || m_cachedEncoderOutputs.size() <= static_cast<size_t>(requiredMaxIndex)) {
        std::cerr << "[ERROR] inferSingleFrame => encoder outputs are not cached.\n";
        return Image<float>();
    }

    if (promptsEmpty(prompts)) {
        return Image<float>(originalImageSize.width, originalImageSize.height, 1);
    }

    try {
        PreparedSAM3MaskPrompt preparedMask;
        bool hasPreparedMask = false;
        if (hasMaskPrompt(prompts)) {
            if (!prepareMaskPrompt(prompts, originalImageSize, &preparedMask)) {
                return Image<float>(originalImageSize.width, originalImageSize.height, 1);
            }
            hasPreparedMask = true;
        }

        const bool useMaskDecoder = hasPreparedMask && m_imageMaskDecoderSession;
        const bool decoderUsesTrackerIo =
            useMaskDecoder ? m_imageMaskDecoderUsesTrackerIo : m_imageDecoderUsesTrackerIo;
        const std::vector<SAM3Node>& decoderInputNodes =
            useMaskDecoder ? m_imageMaskDecoderInputNodes : m_imageDecoderInputNodes;
        const std::vector<const char*>& decoderInputNames =
            useMaskDecoder ? m_imageMaskDecoderInputNames : m_imageDecoderInputNames;
        const std::vector<const char*>& decoderOutputNames =
            useMaskDecoder ? m_imageMaskDecoderOutputNames : m_imageDecoderOutputNames;
        const int decoderPredMasksIndex =
            useMaskDecoder ? m_imageMaskDecoderPredMasksIndex : m_imageDecoderPredMasksIndex;
        const int decoderIouScoresIndex =
            useMaskDecoder ? m_imageMaskDecoderIouScoresIndex : m_imageDecoderIouScoresIndex;
        Ort::Session* decoderSession =
            useMaskDecoder ? m_imageMaskDecoderSession.get() : m_imageDecoderSession.get();

        std::vector<float> promptPoints;
        std::vector<int64_t> promptLabels;
        std::vector<float> promptBoxes;
        std::vector<float> trackerPointCoords;
        std::vector<int32_t> trackerPointLabels;
        if (decoderUsesTrackerIo) {
            buildTrackerPromptInputs(
                prompts,
                originalImageSize,
                &trackerPointCoords,
                &trackerPointLabels);
        } else {
            buildImagePromptInputs(
                prompts,
                originalImageSize,
                &promptPoints,
                &promptLabels,
                &promptBoxes);
        }

        const int64_t numPoints = static_cast<int64_t>(promptLabels.size());
        const int64_t numBoxes = static_cast<int64_t>(promptBoxes.size() / 4);
        const int64_t trackerNumPoints = static_cast<int64_t>(trackerPointLabels.size());
        const std::vector<int64_t> pointsShape = {1, 1, numPoints, 2};
        const std::vector<int64_t> labelsShape = {1, 1, numPoints};
        const std::vector<int64_t> boxesShape = {1, numBoxes, 4};
        const std::vector<int64_t> trackerPointsShape = {1, trackerNumPoints, 2};
        const std::vector<int64_t> trackerLabelsShape = {1, trackerNumPoints};

        std::vector<float> emptyMaskLogits;
        std::vector<int64_t> maskLogitsShape;
        if (useMaskDecoder) {
            if (hasPreparedMask) {
                maskLogitsShape = preparedMask.maskPromptShape;
            } else {
                const SAM3Size promptSize = promptMaskInputSize(getInputSize());
                maskLogitsShape = {1, 1, promptSize.height, promptSize.width};
                emptyMaskLogits.assign(
                    static_cast<std::size_t>(promptSize.width) * static_cast<std::size_t>(promptSize.height),
                    0.0f);
            }
        }

        const std::vector<int64_t> emb0Shape =
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)]
                .GetTensorTypeAndShapeInfo()
                .GetShape();
        const std::vector<int64_t> emb1Shape =
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)]
                .GetTensorTypeAndShapeInfo()
                .GetShape();
        const std::vector<int64_t> emb2Shape =
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)]
                .GetTensorTypeAndShapeInfo()
                .GetShape();
        const std::vector<float>* imageEmbed = nullptr;
        if (decoderUsesTrackerIo) {
            imageEmbed = &buildNoMemoryImageEmbedding(
                m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)],
                emb2Shape);
        }

        std::vector<Ort::Value> inputs;
        inputs.reserve(decoderInputNodes.size());
        for (const SAM3Node& inputNode : decoderInputNodes) {
            const std::string name = lowerCopy(inputNode.name);
            if (name.find("input_points") != std::string::npos) {
                inputs.push_back(createTensor<float>(m_memoryInfo, promptPoints, pointsShape));
            } else if (name.find("input_labels") != std::string::npos) {
                inputs.push_back(createTensor<int64_t>(m_memoryInfo, promptLabels, labelsShape));
            } else if (name.find("input_boxes") != std::string::npos) {
                inputs.push_back(createTensor<float>(m_memoryInfo, promptBoxes, boxesShape));
            } else if (name.find("point_coords") != std::string::npos) {
                inputs.push_back(createTensor<float>(m_memoryInfo, trackerPointCoords, trackerPointsShape));
            } else if (name.find("point_labels") != std::string::npos) {
                inputs.push_back(createTensor<int32_t>(m_memoryInfo, trackerPointLabels, trackerLabelsShape));
            } else if (name.find("mask_input") != std::string::npos) {
                const std::vector<float>& maskLogits =
                    hasPreparedMask ? preparedMask.maskPrompt : emptyMaskLogits;
                inputs.push_back(createTensor<float>(m_memoryInfo, maskLogits, maskLogitsShape));
            } else if (name.find("image_embeddings.0") != std::string::npos
                       || name.find("high_res_feats_0") != std::string::npos) {
                inputs.push_back(createTensorView<float>(
                    m_memoryInfo,
                    m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)].GetTensorMutableData<float>(),
                    emb0Shape));
            } else if (name.find("image_embeddings.1") != std::string::npos
                       || name.find("high_res_feats_1") != std::string::npos) {
                inputs.push_back(createTensorView<float>(
                    m_memoryInfo,
                    m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)].GetTensorMutableData<float>(),
                    emb1Shape));
            } else if (name.find("image_embeddings.2") != std::string::npos) {
                inputs.push_back(createTensorView<float>(
                    m_memoryInfo,
                    m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)].GetTensorMutableData<float>(),
                    emb2Shape));
            } else if (name == "image_embed" && imageEmbed) {
                inputs.push_back(createTensor<float>(m_memoryInfo, *imageEmbed, emb2Shape));
            } else {
                throw std::runtime_error("Unsupported image decoder input: " + inputNode.name);
            }
        }

        auto result = runSession(
            decoderSession,
            decoderInputNames,
            decoderOutputNames,
            inputs,
            "imageDecoder");
        if (result.index() == 1) {
            std::cerr << std::get<std::string>(result) << '\n';
            return Image<float>();
        }

        auto outputs = std::move(std::get<0>(result));
        if (outputs.size() <= static_cast<size_t>(std::max(decoderPredMasksIndex, decoderIouScoresIndex))) {
            std::cerr << "[ERROR] Image decoder returned insufficient outputs.\n";
            return Image<float>();
        }

        const Ort::Value& predMasks = outputs[static_cast<size_t>(decoderPredMasksIndex)];
        const Ort::Value& iouScores = outputs[static_cast<size_t>(decoderIouScoresIndex)];
        return createImageMaskFromDecoderOutput(predMasks, iouScores, originalImageSize);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] inferSingleFrame => " << error.what() << '\n';
        return Image<float>();
    }
}

} // namespace smseg_sam3
