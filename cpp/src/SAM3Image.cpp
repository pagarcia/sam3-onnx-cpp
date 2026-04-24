#include "SAM3.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "CVHelpers.h"

namespace {

constexpr float kSam3Mean = 0.5f;
constexpr float kSam3Std = 0.5f;

} // namespace

bool SAM3::promptsEmpty(const SAM3Prompts& prompts) const
{
    return prompts.points.empty() && prompts.rects.empty();
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
}

Image<float> SAM3::createImageMaskFromLogits(const float* logits,
                                             int maskWidth,
                                             int maskHeight,
                                             const SAM3Size& originalImageSize) const
{
    if (!logits || maskWidth <= 0 || maskHeight <= 0) {
        return Image<float>();
    }

    cv::Mat lowRes(maskHeight, maskWidth, CV_32FC1, const_cast<float*>(logits));
    cv::Mat upsampledInput;
    const SAM3Size inputSize = getInputSize();
    cv::resize(
        lowRes,
        upsampledInput,
        cv::Size(inputSize.width, inputSize.height),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    cv::Mat upsampledOriginal;
    cv::resize(
        upsampledInput,
        upsampledOriginal,
        cv::Size(originalImageSize.width, originalImageSize.height),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    cv::Mat binary;
    cv::threshold(upsampledOriginal, binary, 0.0, 1.0, cv::THRESH_BINARY);
    return CVHelpers::cvMatToImage<float>(binary);
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

    std::vector<float> promptPoints;
    std::vector<int64_t> promptLabels;
    std::vector<float> promptBoxes;
    buildImagePromptInputs(prompts, originalImageSize, &promptPoints, &promptLabels, &promptBoxes);

    const int64_t numPoints = static_cast<int64_t>(promptLabels.size());
    const int64_t numBoxes = static_cast<int64_t>(promptBoxes.size() / 4);
    const std::vector<int64_t> pointsShape = {1, 1, numPoints, 2};
    const std::vector<int64_t> labelsShape = {1, 1, numPoints};
    const std::vector<int64_t> boxesShape = {1, numBoxes, 4};

    try {
        std::vector<Ort::Value> inputs;
        inputs.reserve(m_imageDecoderInputNodes.size());
        inputs.push_back(createTensor<float>(m_memoryInfo, promptPoints, pointsShape));
        inputs.push_back(createTensor<int64_t>(m_memoryInfo, promptLabels, labelsShape));
        inputs.push_back(createTensor<float>(m_memoryInfo, promptBoxes, boxesShape));

        const auto emb0Shape = m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)]
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape();
        const auto emb1Shape = m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)]
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape();
        const auto emb2Shape = m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)]
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape();
        inputs.push_back(createTensorView<float>(
            m_memoryInfo,
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)].GetTensorMutableData<float>(),
            emb0Shape));
        inputs.push_back(createTensorView<float>(
            m_memoryInfo,
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)].GetTensorMutableData<float>(),
            emb1Shape));
        inputs.push_back(createTensorView<float>(
            m_memoryInfo,
            m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)].GetTensorMutableData<float>(),
            emb2Shape));

        auto result = runSession(
            m_imageDecoderSession.get(),
            m_imageDecoderInputNames,
            m_imageDecoderOutputNames,
            inputs,
            "imageDecoder");
        if (result.index() == 1) {
            std::cerr << std::get<std::string>(result) << '\n';
            return Image<float>();
        }

        auto outputs = std::move(std::get<0>(result));
        if (outputs.size() <= static_cast<size_t>(std::max(m_imageDecoderPredMasksIndex, m_imageDecoderIouScoresIndex))) {
            std::cerr << "[ERROR] Image decoder returned insufficient outputs.\n";
            return Image<float>();
        }

        const Ort::Value& predMasks = outputs[static_cast<size_t>(m_imageDecoderPredMasksIndex)];
        const Ort::Value& iouScores = outputs[static_cast<size_t>(m_imageDecoderIouScoresIndex)];
        const auto maskShape = predMasks.GetTensorTypeAndShapeInfo().GetShape();
        const auto scoreShape = iouScores.GetTensorTypeAndShapeInfo().GetShape();
        if (maskShape.size() < 5 || scoreShape.size() < 3) {
            std::cerr << "[ERROR] Unexpected image decoder output shapes.\n";
            return Image<float>();
        }

        const int64_t numMasks = maskShape[2];
        const int64_t maskHeight = maskShape[3];
        const int64_t maskWidth = maskShape[4];
        const float* maskData = predMasks.GetTensorData<float>();
        const float* scoreData = iouScores.GetTensorData<float>();
        if (!maskData || !scoreData || numMasks <= 0 || maskHeight <= 0 || maskWidth <= 0) {
            std::cerr << "[ERROR] Decoder returned empty mask data.\n";
            return Image<float>();
        }

        int bestMaskIndex = 0;
        float bestScore = scoreData[0];
        for (int64_t index = 1; index < numMasks; ++index) {
            if (scoreData[index] > bestScore) {
                bestScore = scoreData[index];
                bestMaskIndex = static_cast<int>(index);
            }
        }

        const size_t maskPlaneSize = static_cast<size_t>(maskHeight * maskWidth);
        return createImageMaskFromLogits(
            maskData + static_cast<size_t>(bestMaskIndex) * maskPlaneSize,
            static_cast<int>(maskWidth),
            static_cast<int>(maskHeight),
            originalImageSize);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] inferSingleFrame => " << error.what() << '\n';
        return Image<float>();
    }
}
