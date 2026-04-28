#include "ArtifactResolver.h"
#include "CVHelpers.h"
#include "SAM3.h"
#include "openFileDialog.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kJumpFrames = 10;
const char* kWindowName = "SAM3 Video Anchors";

enum class PromptMode { SeedPoints, BoundingBox };

struct AnchorEditorState {
    SAM3* sam = nullptr;
    cv::VideoCapture* capture = nullptr;
    PromptMode mode = PromptMode::SeedPoints;
    int totalFrames = 0;
    int currentFrameIndex = 0;
    cv::Mat currentFrame;
    cv::Mat displayFrame;
    SAM3Size originalSize;
    std::map<int, SAM3Prompts> anchors;
    std::map<int, CachedEncoderOutputs> anchorEncoderCaches;
    std::vector<SAM3Point> currentPoints;
    std::vector<int> currentLabels;
    bool drawing = false;
    bool hasFinalRect = false;
    SAM3Rect rect;
};

Image<float> normalizeSam3Image(const cv::Mat& image)
{
    return CVHelpers::normalizeRGB(
        image,
        255.0f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f);
}

cv::Mat overlayMask(const cv::Mat& image, const cv::Mat& maskGray)
{
    cv::Mat overlay = image.clone();
    cv::Mat green(image.size(), image.type(), cv::Scalar(0, 255, 0));
    green.copyTo(overlay, maskGray);

    cv::Mat blended;
    cv::addWeighted(image, 0.7, overlay, 0.3, 0.0, blended);
    return blended;
}

bool parsePointsSpec(const std::string& text,
                     std::vector<SAM3Point>* pointsOut,
                     std::vector<int>* labelsOut)
{
    pointsOut->clear();
    labelsOut->clear();
    if (text.empty()) {
        return true;
    }

    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ';')) {
        if (item.empty()) {
            continue;
        }

        std::stringstream itemStream(item);
        std::string xText;
        std::string yText;
        std::string labelText;
        if (!std::getline(itemStream, xText, ',')
            || !std::getline(itemStream, yText, ',')
            || !std::getline(itemStream, labelText, ',')) {
            return false;
        }

        pointsOut->push_back(SAM3Point(std::stoi(xText), std::stoi(yText)));
        labelsOut->push_back(std::stoi(labelText));
    }
    return true;
}

bool parseBoxSpec(const std::string& text, SAM3Rect* rectOut)
{
    if (text.empty()) {
        return false;
    }

    std::stringstream stream(text);
    std::string x1Text;
    std::string y1Text;
    std::string x2Text;
    std::string y2Text;
    if (!std::getline(stream, x1Text, ',')
        || !std::getline(stream, y1Text, ',')
        || !std::getline(stream, x2Text, ',')
        || !std::getline(stream, y2Text, ',')) {
        return false;
    }

    const int x1 = std::stoi(x1Text);
    const int y1 = std::stoi(y1Text);
    const int x2 = std::stoi(x2Text);
    const int y2 = std::stoi(y2Text);
    *rectOut = SAM3Rect(x1, y1, x2 - x1, y2 - y1);
    return true;
}

bool parseMaskPromptStrategy(const std::string& text, SAM3MaskPromptStrategy* strategyOut)
{
    if (text == "box") {
        *strategyOut = SAM3MaskPromptStrategy::Box;
        return true;
    }
    if (text == "point") {
        *strategyOut = SAM3MaskPromptStrategy::Point;
        return true;
    }
    return false;
}

bool loadMaskPromptImage(const std::string& path, Image<float>* maskOut)
{
    if (path.empty() || !maskOut) {
        return false;
    }

    const cv::Mat maskGray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (maskGray.empty()) {
        std::cerr << "[ERROR] Could not read mask prompt " << path << '\n';
        return false;
    }

    cv::Mat maskFloat;
    maskGray.convertTo(maskFloat, CV_32FC1, 1.0 / 255.0);
    *maskOut = CVHelpers::cvMatToImage<float>(maskFloat);
    return true;
}

SAM3Rect normalizedRect(const SAM3Rect& rect)
{
    SAM3Rect normalized = rect;
    if (normalized.width < 0) {
        normalized.x += normalized.width;
        normalized.width = -normalized.width;
    }
    if (normalized.height < 0) {
        normalized.y += normalized.height;
        normalized.height = -normalized.height;
    }
    return normalized;
}

int clampFrameIndex(int frameIndex, int totalFrames)
{
    if (totalFrames <= 0) {
        return 0;
    }
    return std::max(0, std::min(frameIndex, totalFrames - 1));
}

int resolveFrameCount(cv::VideoCapture& capture, int maxFrames)
{
    int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    if (totalFrames <= 0) {
        return maxFrames > 0 ? maxFrames : 0;
    }
    if (maxFrames > 0) {
        totalFrames = std::min(totalFrames, maxFrames);
    }
    return std::max(totalFrames, 1);
}

bool loadFrameAt(cv::VideoCapture& capture, int frameIndex, cv::Mat* frameOut)
{
    capture.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty()) {
        return false;
    }
    *frameOut = frame;
    return true;
}

SAM3Prompts buildCurrentPrompts(const AnchorEditorState& state)
{
    SAM3Prompts prompts;
    if (state.mode == PromptMode::SeedPoints) {
        prompts.points = state.currentPoints;
        prompts.pointLabels = state.currentLabels;
    } else if (state.hasFinalRect) {
        prompts.rects.push_back(normalizedRect(state.rect));
    }
    return prompts;
}

void clearCurrentPrompt(AnchorEditorState* state)
{
    state->currentPoints.clear();
    state->currentLabels.clear();
    state->drawing = false;
    state->hasFinalRect = false;
    state->rect = SAM3Rect();
}

void loadCurrentPromptFromAnchor(AnchorEditorState* state)
{
    clearCurrentPrompt(state);
    const auto it = state->anchors.find(state->currentFrameIndex);
    if (it == state->anchors.end()) {
        return;
    }

    if (state->mode == PromptMode::SeedPoints) {
        state->currentPoints = it->second.points;
        state->currentLabels = it->second.pointLabels;
        return;
    }

    if (!it->second.rects.empty()) {
        state->rect = normalizedRect(it->second.rects.front());
        state->hasFinalRect = true;
    }
}

void storeCurrentPromptToAnchor(AnchorEditorState* state)
{
    const SAM3Prompts prompts = buildCurrentPrompts(*state);
    const bool hasPrompt =
        (!prompts.points.empty() && prompts.points.size() == prompts.pointLabels.size())
        || !prompts.rects.empty();
    if (!hasPrompt) {
        state->anchors.erase(state->currentFrameIndex);
        state->anchorEncoderCaches.erase(state->currentFrameIndex);
        return;
    }

    state->anchors[state->currentFrameIndex] = prompts;
    CachedEncoderOutputs cachedOutputs;
    if (state->sam->captureCachedEncoderOutputs(&cachedOutputs)) {
        state->anchorEncoderCaches[state->currentFrameIndex] = std::move(cachedOutputs);
    }
}

bool preprocessCurrentFrame(AnchorEditorState* state)
{
    state->originalSize = SAM3Size(state->currentFrame.cols, state->currentFrame.rows);
    const auto cacheIt = state->anchorEncoderCaches.find(state->currentFrameIndex);
    if (cacheIt != state->anchorEncoderCaches.end() && state->sam->restoreCachedEncoderOutputs(cacheIt->second)) {
        return true;
    }
    return state->sam->preprocessImage(normalizeSam3Image(state->currentFrame));
}

void drawHud(cv::Mat* image, int frameIndex, int totalFrames, size_t anchorCount, PromptMode mode)
{
    std::vector<std::string> lines;
    lines.push_back(
        "Frame " + std::to_string(frameIndex + 1) + "/" + std::to_string(totalFrames)
        + " | Anchors: " + std::to_string(anchorCount));
    lines.push_back("Prompt: " + std::string(mode == PromptMode::SeedPoints ? "seed_points" : "bounding_box"));
    lines.push_back("A/D: +/-1 frame | J/L: +/-10 frames");
    lines.push_back("Enter/Space: run video | Esc/Q: finish | C: clear current frame");

    int y = 28;
    for (const auto& line : lines) {
        cv::putText(*image, line, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(10, 10, 10), 4, cv::LINE_AA);
        cv::putText(*image, line, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        y += 28;
    }
}

void renderAnchorEditor(AnchorEditorState* state)
{
    state->displayFrame = state->currentFrame.clone();

    const SAM3Prompts prompts = buildCurrentPrompts(*state);
    const bool shouldRun = !prompts.points.empty() || !prompts.rects.empty();
    if (shouldRun) {
        const Image<float> mask = state->sam->previewConditioningFrame(state->originalSize, prompts);
        state->displayFrame = overlayMask(
            state->displayFrame,
            CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0));
    }

    if (state->mode == PromptMode::SeedPoints) {
        for (size_t index = 0; index < state->currentPoints.size(); ++index) {
            const cv::Scalar color =
                state->currentLabels[index] == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            cv::circle(
                state->displayFrame,
                cv::Point(state->currentPoints[index].x, state->currentPoints[index].y),
                5,
                color,
                -1);
        }
    } else if (state->drawing || state->hasFinalRect) {
        const SAM3Rect rect = normalizedRect(state->rect);
        cv::rectangle(
            state->displayFrame,
            cv::Rect(rect.x, rect.y, rect.width, rect.height),
            cv::Scalar(0, 255, 255),
            2);
    }

    drawHud(&state->displayFrame, state->currentFrameIndex, state->totalFrames, state->anchors.size(), state->mode);
    cv::imshow(kWindowName, state->displayFrame);
}

bool gotoFrame(AnchorEditorState* state, int frameIndex)
{
    storeCurrentPromptToAnchor(state);
    state->currentFrameIndex = clampFrameIndex(frameIndex, state->totalFrames);
    if (!loadFrameAt(*state->capture, state->currentFrameIndex, &state->currentFrame)) {
        return false;
    }
    if (!preprocessCurrentFrame(state)) {
        return false;
    }
    loadCurrentPromptFromAnchor(state);
    renderAnchorEditor(state);
    return true;
}

SAM3Point clampPointToFrame(const AnchorEditorState& state, int x, int y)
{
    return SAM3Point(
        std::max(0, std::min(x, state.currentFrame.cols - 1)),
        std::max(0, std::min(y, state.currentFrame.rows - 1)));
}

void onMouseAnchorEditor(int event, int x, int y, int, void* userData)
{
    auto* state = static_cast<AnchorEditorState*>(userData);
    if (!state || state->currentFrame.empty()) {
        return;
    }

    if (state->mode == PromptMode::SeedPoints) {
        if (event == cv::EVENT_MBUTTONDOWN) {
            clearCurrentPrompt(state);
            storeCurrentPromptToAnchor(state);
            renderAnchorEditor(state);
            return;
        }
        if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_RBUTTONDOWN) {
            return;
        }

        const SAM3Point point = clampPointToFrame(*state, x, y);
        state->currentPoints.push_back(point);
        state->currentLabels.push_back(event == cv::EVENT_LBUTTONDOWN ? 1 : 0);
        storeCurrentPromptToAnchor(state);
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN) {
        clearCurrentPrompt(state);
        storeCurrentPromptToAnchor(state);
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        const SAM3Point point = clampPointToFrame(*state, x, y);
        state->drawing = true;
        state->hasFinalRect = false;
        state->rect = SAM3Rect(point.x, point.y, 0, 0);
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_MOUSEMOVE && state->drawing) {
        const SAM3Point point = clampPointToFrame(*state, x, y);
        state->rect.width = point.x - state->rect.x;
        state->rect.height = point.y - state->rect.y;
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_LBUTTONUP && state->drawing) {
        const SAM3Point point = clampPointToFrame(*state, x, y);
        state->drawing = false;
        state->rect.width = point.x - state->rect.x;
        state->rect.height = point.y - state->rect.y;
        state->rect = normalizedRect(state->rect);
        state->hasFinalRect = state->rect.width > 1 && state->rect.height > 1;
        if (!state->hasFinalRect) {
            state->rect = SAM3Rect();
        }
        storeCurrentPromptToAnchor(state);
        renderAnchorEditor(state);
    }
}

bool collectAnchorPrompts(SAM3* previewSam,
                          const std::string& videoPath,
                          PromptMode mode,
                          int maxFrames,
                          std::map<int, SAM3Prompts>* anchorsOut,
                          std::map<int, CachedEncoderOutputs>* anchorCachesOut,
                          int* totalFramesOut)
{
    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] Could not open the input video.\n";
        return false;
    }

    const int totalFrames = resolveFrameCount(capture, maxFrames);
    if (totalFrames <= 0) {
        std::cerr << "[ERROR] Could not determine the frame count.\n";
        return false;
    }

    AnchorEditorState state;
    state.sam = previewSam;
    state.capture = &capture;
    state.mode = mode;
    state.totalFrames = totalFrames;

    if (!loadFrameAt(capture, 0, &state.currentFrame)) {
        return false;
    }
    if (!preprocessCurrentFrame(&state)) {
        return false;
    }

    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(kWindowName, onMouseAnchorEditor, &state);
    renderAnchorEditor(&state);

    std::cout << "[INFO] Anchor editor ready. A/D = +/-1 frame, J/L = +/-10, Enter/Space = run.\n";
    while (true) {
        const int key = cv::waitKey(20) & 0xFF;
        if (key == 13 || key == 10 || key == 32 || key == 27 || key == 'q' || key == 'Q') {
            storeCurrentPromptToAnchor(&state);
            break;
        }
        if (key == 'a' || key == 'A') {
            gotoFrame(&state, state.currentFrameIndex - 1);
        } else if (key == 'd' || key == 'D') {
            gotoFrame(&state, state.currentFrameIndex + 1);
        } else if (key == 'j' || key == 'J') {
            gotoFrame(&state, state.currentFrameIndex - kJumpFrames);
        } else if (key == 'l' || key == 'L') {
            gotoFrame(&state, state.currentFrameIndex + kJumpFrames);
        } else if (key == 'c' || key == 'C') {
            clearCurrentPrompt(&state);
            storeCurrentPromptToAnchor(&state);
            renderAnchorEditor(&state);
        }
    }

    cv::destroyAllWindows();
    *anchorsOut = std::move(state.anchors);
    *anchorCachesOut = std::move(state.anchorEncoderCaches);
    *totalFramesOut = totalFrames;
    return true;
}

std::string resolveVideoPath(const std::string& videoPath)
{
    if (!videoPath.empty()) {
        return videoPath;
    }

    const wchar_t* filter = L"Video\0*.mp4;*.mkv;*.avi;*.mov\0All\0*.*\0";
    return openFileDialog(filter, L"Select a video");
}

std::string defaultOutputPath(const std::string& videoPath,
                              const ArtifactResolver::VideoRuntimeSelection& selection)
{
    const std::filesystem::path path(videoPath);
    const std::string stem = path.stem().string();
    const std::string suffix = "_" + selection.graphProfile + "_" + selection.precision + "_sam3_overlay.avi";
    return (path.parent_path() / (stem + suffix)).string();
}

std::string normalizeDeviceArgument(const std::string& value)
{
    const std::string lowered = ArtifactResolver::lowerCopy(value);
    if (lowered.empty()) {
        return std::string();
    }
    if (lowered == "cpu") {
        return "cpu";
    }
    if (lowered == "cuda" || lowered == "gpu") {
        return "cuda:0";
    }
    if (lowered.rfind("cuda:", 0) == 0) {
        return lowered;
    }
    if (lowered == "dml" || lowered == "directml") {
        return "dml:0";
    }
    if (lowered.rfind("dml:", 0) == 0) {
        return lowered;
    }
    return std::string();
}

void printRuntimeSelection(const char* label,
                           const ArtifactResolver::VideoRuntimeSelection& selection,
                           const std::string& device,
                           int threads)
{
    std::cout << "[INFO] " << label << '\n'
              << "       encoder     = " << selection.encoderPath << '\n'
              << "       decoder     = " << selection.decoderPath << '\n'
              << "       memAttn     = " << selection.memoryAttentionPath << '\n'
              << "       memEnc      = " << selection.memoryEncoderPath << '\n'
              << "       constants   = " << selection.constantsPath << '\n'
              << "       precision   = " << selection.precision << '\n'
              << "       graph       = " << selection.graphProfile << '\n'
              << "       device      = " << device << '\n'
              << "       threads     = " << threads << '\n';
    if (device == "cpu" && ArtifactResolver::lowerCopy(selection.encoderPath).find("fp16") != std::string::npos) {
        std::cout
            << "[WARN] CPU runtime is using the fp16 vision encoder fallback. "
            << "A fp32 or int8 encoder artifact will be much faster on CPU.\n";
    }
}

} // namespace

int runOnnxTestVideo(int argc, char** argv)
{
    std::string encoderPath;
    std::string decoderPath;
    std::string memoryAttentionPath;
    std::string memoryEncoderPath;
    std::string constantsPath;
    std::string requestedDevice;
    std::string videoPath;
    std::string pointsSpec;
    std::string boxSpec;
    std::string maskPath;
    std::string outputPath;
    int hardwareThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (hardwareThreads <= 0) {
        hardwareThreads = 4;
    }
    int threads = hardwareThreads;
    bool threadsExplicit = false;
    int maxFrames = 0;
    PromptMode promptMode = PromptMode::SeedPoints;
    SAM3MaskPromptStrategy maskPromptStrategy = SAM3MaskPromptStrategy::Box;

    for (int index = 2; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--encoder" && index + 1 < argc) {
            encoderPath = argv[++index];
        } else if (arg == "--decoder" && index + 1 < argc) {
            decoderPath = argv[++index];
        } else if (arg == "--memattn" && index + 1 < argc) {
            memoryAttentionPath = argv[++index];
        } else if (arg == "--memenc" && index + 1 < argc) {
            memoryEncoderPath = argv[++index];
        } else if (arg == "--constants" && index + 1 < argc) {
            constantsPath = argv[++index];
        } else if (arg == "--device" && index + 1 < argc) {
            requestedDevice = argv[++index];
        } else if (arg == "--video" && index + 1 < argc) {
            videoPath = argv[++index];
        } else if (arg == "--threads" && index + 1 < argc) {
            threads = std::stoi(argv[++index]);
            threadsExplicit = true;
        } else if (arg == "--max_frames" && index + 1 < argc) {
            maxFrames = std::stoi(argv[++index]);
        } else if (arg == "--prompt" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (value == "seed_points") {
                promptMode = PromptMode::SeedPoints;
            } else if (value == "bounding_box") {
                promptMode = PromptMode::BoundingBox;
            } else {
                std::cerr << "[ERROR] --prompt must be seed_points|bounding_box\n";
                return 1;
            }
        } else if (arg == "--points" && index + 1 < argc) {
            pointsSpec = argv[++index];
        } else if (arg == "--box" && index + 1 < argc) {
            boxSpec = argv[++index];
        } else if (arg == "--mask" && index + 1 < argc) {
            maskPath = argv[++index];
        } else if (arg == "--mask_prompt_strategy" && index + 1 < argc) {
            if (!parseMaskPromptStrategy(argv[++index], &maskPromptStrategy)) {
                std::cerr << "[ERROR] --mask_prompt_strategy must be box|point\n";
                return 1;
            }
        } else if (arg == "--output" && index + 1 < argc) {
            outputPath = argv[++index];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: Segment --onnx_test_video [options]\n\n"
                << "Options:\n"
                << "  --video path                optional video path\n"
                << "  --encoder path              optional encoder override\n"
                << "  --decoder path              optional tracker decoder override\n"
                << "  --memattn path              optional memory attention override\n"
                << "  --memenc path               optional memory encoder override\n"
                << "  --constants path            optional NPZ constants override\n"
                << "  --device cpu|cuda|cuda:N|dml|dml:N    optional runtime device override\n"
                << "  --prompt seed_points|bounding_box\n"
                << "  --points x,y,label;...      noninteractive frame-0 prompt\n"
                << "  --box x1,y1,x2,y2           noninteractive frame-0 prompt\n"
                << "  --mask path                 noninteractive dense mask prompt image for frame 0\n"
                << "  --mask_prompt_strategy box|point\n"
                << "  --max_frames N              frame limit\n"
                << "  --output path               output overlay video path\n";
            return 0;
        }
    }

    if (!pointsSpec.empty() && !boxSpec.empty()) {
        std::cerr << "[ERROR] Use either --points or --box, not both.\n";
        return 1;
    }

    videoPath = resolveVideoPath(videoPath);
    if (videoPath.empty()) {
        std::cerr << "[ERROR] No video selected.\n";
        return 1;
    }

    const bool deviceExplicit = !requestedDevice.empty();
    std::string device;
    if (deviceExplicit) {
        device = normalizeDeviceArgument(requestedDevice);
        if (device.empty()) {
            std::cerr << "[ERROR] --device must be cpu|cuda|cuda:N|dml|dml:N\n";
            return 1;
        }
    } else {
        const bool forceCpu = ArtifactResolver::isLowCostCpuProfile();
        const bool cudaAvailable = !forceCpu && SAM3::hasCudaDriver();
        const bool dmlAvailable = !forceCpu && SAM3::hasDirectMLProvider();
        device = cudaAvailable ? "cuda:0" : (dmlAvailable ? "dml:0" : "cpu");
    }
    if (!threadsExplicit) {
        threads = ArtifactResolver::preferredRuntimeThreads(hardwareThreads, device);
    }

    auto previewSelection = ArtifactResolver::resolveVideoRuntimePaths(
        encoderPath,
        decoderPath,
        memoryAttentionPath,
        memoryEncoderPath,
        constantsPath,
        device,
        1);

    SAM3 previewSam;
    auto initializePreviewOnDevice = [&](const std::string& initDevice) -> bool {
        const int initThreads = threadsExplicit
            ? threads
            : ArtifactResolver::preferredRuntimeThreads(hardwareThreads, initDevice);
        const auto initSelection = ArtifactResolver::resolveVideoRuntimePaths(
            encoderPath,
            decoderPath,
            memoryAttentionPath,
            memoryEncoderPath,
            constantsPath,
            initDevice,
            1);
        printRuntimeSelection("Preview tracker bundle", initSelection, initDevice, initThreads);
        if (!previewSam.initializeVideo(
            initSelection.encoderPath,
            initSelection.decoderPath,
            initSelection.memoryAttentionPath,
            initSelection.memoryEncoderPath,
            initSelection.constantsPath,
            initThreads,
            initDevice)) {
            std::cerr << "[ERROR] Failed to initialize preview tracker runtime on " << initDevice << ".\n";
            return false;
        }
        previewSelection = initSelection;
        device = initDevice;
        if (!threadsExplicit) {
            threads = initThreads;
        }
        return true;
    };

    if (!initializePreviewOnDevice(device)) {
        if (deviceExplicit || device == "cpu") {
            return 1;
        }
        std::cerr << "[WARN] Falling back to CPU runtime for preview.\n";
        if (!initializePreviewOnDevice("cpu")) {
            return 1;
        }
    }

    std::map<int, SAM3Prompts> anchors;
    std::map<int, CachedEncoderOutputs> anchorCaches;
    int totalFrames = 0;

    if (!pointsSpec.empty() || !boxSpec.empty() || !maskPath.empty()) {
        cv::VideoCapture capture(videoPath);
        if (!capture.isOpened()) {
            std::cerr << "[ERROR] Could not open the input video.\n";
            return 1;
        }

        totalFrames = resolveFrameCount(capture, maxFrames);
        cv::Mat firstFrame;
        if (!capture.read(firstFrame) || firstFrame.empty()) {
            std::cerr << "[ERROR] Could not read frame 0.\n";
            return 1;
        }
        if (!previewSam.preprocessImage(normalizeSam3Image(firstFrame))) {
            return 1;
        }

        SAM3Prompts prompts;
        if (!maskPath.empty()) {
            if (!loadMaskPromptImage(maskPath, &prompts.mask)) {
                return 1;
            }
            prompts.maskPromptStrategy = maskPromptStrategy;
        }
        if (!pointsSpec.empty()) {
            if (!parsePointsSpec(pointsSpec, &prompts.points, &prompts.pointLabels)) {
                std::cerr << "[ERROR] Could not parse --points.\n";
                return 1;
            }
        } else if (!boxSpec.empty()) {
            SAM3Rect rect;
            if (!parseBoxSpec(boxSpec, &rect)) {
                std::cerr << "[ERROR] Could not parse --box.\n";
                return 1;
            }
            prompts.rects.push_back(rect);
        }
        anchors[0] = prompts;

        CachedEncoderOutputs cachedOutputs;
        if (previewSam.captureCachedEncoderOutputs(&cachedOutputs)) {
            anchorCaches[0] = std::move(cachedOutputs);
        }
    } else {
        if (!collectAnchorPrompts(
                &previewSam,
                videoPath,
                promptMode,
                maxFrames,
                &anchors,
                &anchorCaches,
                &totalFrames)) {
            return 1;
        }
    }

    if (anchors.empty()) {
        std::cerr << "[ERROR] No anchors were provided.\n";
        return 1;
    }

    auto runtimeSelection = ArtifactResolver::resolveVideoRuntimePaths(
        encoderPath,
        decoderPath,
        memoryAttentionPath,
        memoryEncoderPath,
        constantsPath,
        device,
        anchors.size());

    SAM3 runtimeSam;
    auto initializeRuntimeOnDevice = [&](const std::string& initDevice) -> bool {
        const int initThreads = threadsExplicit
            ? threads
            : ArtifactResolver::preferredRuntimeThreads(hardwareThreads, initDevice);
        const auto initSelection = ArtifactResolver::resolveVideoRuntimePaths(
            encoderPath,
            decoderPath,
            memoryAttentionPath,
            memoryEncoderPath,
            constantsPath,
            initDevice,
            anchors.size());
        printRuntimeSelection("Runtime tracker bundle", initSelection, initDevice, initThreads);
        if (!runtimeSam.initializeVideo(
            initSelection.encoderPath,
            initSelection.decoderPath,
            initSelection.memoryAttentionPath,
            initSelection.memoryEncoderPath,
            initSelection.constantsPath,
            initThreads,
            initDevice)) {
            std::cerr << "[ERROR] Failed to initialize tracking runtime on " << initDevice << ".\n";
            return false;
        }
        runtimeSelection = initSelection;
        device = initDevice;
        if (!threadsExplicit) {
            threads = initThreads;
        }
        return true;
    };

    if (!initializeRuntimeOnDevice(device)) {
        if (deviceExplicit || device == "cpu") {
            return 1;
        }
        std::cerr << "[WARN] Falling back to CPU runtime for tracking.\n";
        if (!initializeRuntimeOnDevice("cpu")) {
            return 1;
        }
    }

    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] Could not reopen the input video.\n";
        return 1;
    }

    const double fpsRaw = capture.get(cv::CAP_PROP_FPS);
    const double fps = fpsRaw > 0.0 ? fpsRaw : 25.0;
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (width <= 0 || height <= 0) {
        std::cerr << "[ERROR] Could not determine the video resolution.\n";
        return 1;
    }

    if (outputPath.empty()) {
        outputPath = defaultOutputPath(videoPath, runtimeSelection);
    }

    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps,
        cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "[ERROR] Could not open the output writer.\n";
        return 1;
    }

    runtimeSam.resetMemory();
    SAM3Prompts emptyPrompts;
    bool activeSegment = false;
    int writtenFrames = 0;

    for (int frameIndex = 0; ; ++frameIndex) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            break;
        }
        if (maxFrames > 0 && frameIndex >= maxFrames) {
            break;
        }
        if (totalFrames > 0 && frameIndex >= totalFrames) {
            break;
        }

        const auto anchorIt = anchors.find(frameIndex);
        if (anchorIt != anchors.end()) {
            runtimeSam.resetMemory();
            activeSegment = true;
            std::cout << "[INFO] Anchor frame " << frameIndex << " resets the tracker state.\n";
        }

        if (!activeSegment) {
            writer << frame;
            ++writtenFrames;
            continue;
        }

        Image<float> mask;
        if (anchorIt != anchors.end()) {
            const auto cacheIt = anchorCaches.find(frameIndex);
            if (cacheIt != anchorCaches.end() && runtimeSam.restoreCachedEncoderOutputs(cacheIt->second)) {
                mask = runtimeSam.inferMultiFrameCached(SAM3Size(frame.cols, frame.rows), anchorIt->second);
            } else {
                mask = runtimeSam.inferMultiFrame(normalizeSam3Image(frame), anchorIt->second);
            }
        } else {
            mask = runtimeSam.inferMultiFrame(normalizeSam3Image(frame), emptyPrompts);
        }

        writer << overlayMask(frame, CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0));
        ++writtenFrames;
    }

    writer.release();
    capture.release();
    std::cout << "[INFO] Saved " << outputPath << " (" << writtenFrames << " frames)\n";
    return 0;
}
