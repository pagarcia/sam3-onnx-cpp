#include "ArtifactResolver.h"
#include "CVHelpers.h"
#include "SAM3.h"
#include "openFileDialog.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

enum class PromptMode { SeedPoints, BoundingBox };

struct AppState {
    SAM3* sam = nullptr;
    cv::Mat original;
    cv::Mat display;
    SAM3Size originalSize;
    PromptMode mode = PromptMode::SeedPoints;
    std::vector<SAM3Point> points;
    std::vector<int> labels;
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

SAM3Prompts buildPrompts(const AppState& state)
{
    SAM3Prompts prompts;
    if (state.mode == PromptMode::SeedPoints) {
        prompts.points = state.points;
        prompts.pointLabels = state.labels;
    } else if (state.hasFinalRect) {
        prompts.rects.push_back(normalizedRect(state.rect));
    }
    return prompts;
}

void updateDisplay(AppState* state, bool forceRunBox = false)
{
    state->display = state->original.clone();

    if (state->mode == PromptMode::BoundingBox && (state->drawing || state->hasFinalRect)) {
        const SAM3Rect rect = normalizedRect(state->rect);
        cv::rectangle(
            state->display,
            cv::Rect(rect.x, rect.y, rect.width, rect.height),
            cv::Scalar(0, 255, 255),
            2);
    }

    bool shouldRun = false;
    if (state->mode == PromptMode::SeedPoints) {
        shouldRun = !state->points.empty();
    } else {
        shouldRun = forceRunBox && state->hasFinalRect;
    }

    if (shouldRun) {
        const auto start = std::chrono::high_resolution_clock::now();
        const Image<float> mask = state->sam->inferSingleFrame(state->originalSize, buildPrompts(*state));
        const double elapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "[INFO] Image decoder => " << elapsedMs << " ms\n";

        state->display = overlayMask(
            state->display,
            CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0));
    }

    if (state->mode == PromptMode::SeedPoints) {
        for (size_t index = 0; index < state->points.size(); ++index) {
            const cv::Scalar color =
                state->labels[index] == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            cv::circle(
                state->display,
                cv::Point(state->points[index].x, state->points[index].y),
                5,
                color,
                -1);
        }
    }

    cv::imshow("SAM3 Image", state->display);
}

void onMouse(int event, int x, int y, int, void* userData)
{
    auto* state = static_cast<AppState*>(userData);
    if (!state) {
        return;
    }

    if (state->mode == PromptMode::SeedPoints) {
        if (event == cv::EVENT_MBUTTONDOWN) {
            state->points.clear();
            state->labels.clear();
            updateDisplay(state);
            return;
        }

        if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_RBUTTONDOWN) {
            return;
        }

        state->points.push_back(SAM3Point(x, y));
        state->labels.push_back(event == cv::EVENT_LBUTTONDOWN ? 1 : 0);
        updateDisplay(state);
        return;
    }

    if (event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN) {
        state->drawing = false;
        state->hasFinalRect = false;
        state->rect = SAM3Rect();
        updateDisplay(state);
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        state->drawing = true;
        state->hasFinalRect = false;
        state->rect = SAM3Rect(x, y, 0, 0);
        updateDisplay(state);
        return;
    }

    if (event == cv::EVENT_MOUSEMOVE && state->drawing) {
        state->rect.width = x - state->rect.x;
        state->rect.height = y - state->rect.y;
        updateDisplay(state);
        return;
    }

    if (event == cv::EVENT_LBUTTONUP && state->drawing) {
        state->drawing = false;
        state->rect.width = x - state->rect.x;
        state->rect.height = y - state->rect.y;
        state->rect = normalizedRect(state->rect);
        state->hasFinalRect = state->rect.width > 1 && state->rect.height > 1;
        updateDisplay(state, true);
    }
}

std::string resolveImagePath(const std::string& imagePath)
{
    if (!imagePath.empty()) {
        return imagePath;
    }

    std::cout << "[INFO] Opening image selector...\n";
    const wchar_t* filter = L"Images\0*.jpg;*.jpeg;*.png;*.bmp\0All\0*.*\0";
    return openFileDialog(filter, L"Select an image");
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

void printRuntimeSelection(const ArtifactResolver::ImageRuntimeSelection& selection,
                           const std::string& device,
                           int threads)
{
    std::cout << "[INFO] encoder=" << selection.encoderPath << '\n'
              << "[INFO] decoder=" << selection.decoderPath << '\n'
              << "[INFO] precision=" << selection.precision << '\n'
              << "[INFO] mode=" << selection.mode << '\n'
              << "[INFO] device=" << device << '\n'
              << "[INFO] threads=" << threads << '\n';
    if (device == "cpu" && ArtifactResolver::lowerCopy(selection.encoderPath).find("fp16") != std::string::npos) {
        std::cout
            << "[WARN] CPU runtime is using the fp16 vision encoder fallback. "
            << "A fp32 or int8 encoder artifact will be much faster on CPU.\n";
    }
}

} // namespace

int runOnnxTestImage(int argc, char** argv)
{
    std::string encoderPath;
    std::string decoderPath;
    std::string requestedDevice;
    std::string imagePath;
    std::string pointsSpec;
    std::string boxSpec;
    std::string maskPath;
    std::string saveOverlayPath;
    int hardwareThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (hardwareThreads <= 0) {
        hardwareThreads = 4;
    }
    int threads = hardwareThreads;
    bool threadsExplicit = false;
    bool noGui = false;
    PromptMode promptMode = PromptMode::SeedPoints;
    SAM3MaskPromptStrategy maskPromptStrategy = SAM3MaskPromptStrategy::Box;

    for (int index = 2; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--encoder" && index + 1 < argc) {
            encoderPath = argv[++index];
        } else if (arg == "--decoder" && index + 1 < argc) {
            decoderPath = argv[++index];
        } else if (arg == "--device" && index + 1 < argc) {
            requestedDevice = argv[++index];
        } else if (arg == "--image" && index + 1 < argc) {
            imagePath = argv[++index];
        } else if (arg == "--threads" && index + 1 < argc) {
            threads = std::stoi(argv[++index]);
            threadsExplicit = true;
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
        } else if (arg == "--save_overlay" && index + 1 < argc) {
            saveOverlayPath = argv[++index];
        } else if (arg == "--no_gui") {
            noGui = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: Segment --onnx_test_image [options]\n\n"
                << "Options:\n"
                << "  --image path                optional image path; opens a file dialog if omitted\n"
                << "  --encoder path              optional image encoder override\n"
                << "  --decoder path              optional prompt decoder override\n"
                << "  --device cpu|cuda|cuda:N|dml|dml:N    optional runtime device override\n"
                << "  --prompt seed_points|bounding_box\n"
                << "  --points x,y,label;...      noninteractive seed-point prompt\n"
                << "  --box x1,y1,x2,y2           noninteractive box prompt\n"
                << "  --mask path                 noninteractive dense mask prompt image\n"
                << "  --mask_prompt_strategy box|point\n"
                << "  --save_overlay path         write the overlay image\n"
                << "  --no_gui                    do not open an OpenCV window\n"
                << "  --threads N                 ORT intra-op threads; 0 keeps ORT default\n";
            return 0;
        }
    }

    if (!pointsSpec.empty() && !boxSpec.empty()) {
        std::cerr << "[ERROR] Use either --points or --box, not both.\n";
        return 1;
    }

    imagePath = resolveImagePath(imagePath);
    if (imagePath.empty()) {
        std::cerr << "[ERROR] No image selected.\n";
        return 1;
    }

    const cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[ERROR] Could not read " << imagePath << '\n';
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

    SAM3 sam;
    auto selection = ArtifactResolver::resolveImageRuntimePaths(encoderPath, decoderPath, device);
    auto initializeOnDevice = [&](const std::string& initDevice) -> bool {
        const int initThreads = threadsExplicit
            ? threads
            : ArtifactResolver::preferredRuntimeThreads(hardwareThreads, initDevice);
        const auto initSelection = ArtifactResolver::resolveImageRuntimePaths(encoderPath, decoderPath, initDevice);
        printRuntimeSelection(initSelection, initDevice, initThreads);
        if (!sam.initializeImage(initSelection.encoderPath, initSelection.decoderPath, initThreads, initDevice)) {
            return false;
        }
        selection = initSelection;
        device = initDevice;
        if (!threadsExplicit) {
            threads = initThreads;
        }
        return true;
    };

    if (!initializeOnDevice(device)) {
        if (!deviceExplicit && device != "cpu") {
            std::cerr << "[WARN] Falling back to CPU runtime.\n";
            if (!initializeOnDevice("cpu")) {
                return 1;
            }
        } else {
            return 1;
        }
    }

    const auto encoderStart = std::chrono::high_resolution_clock::now();
    if (!sam.preprocessImage(normalizeSam3Image(image))) {
        return 1;
    }
    const double encoderElapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - encoderStart).count();
    std::cout << "[INFO] Encoder => " << encoderElapsedMs << " ms\n";

    SAM3Prompts nonInteractivePrompts;
    if (!maskPath.empty()) {
        if (!loadMaskPromptImage(maskPath, &nonInteractivePrompts.mask)) {
            return 1;
        }
        nonInteractivePrompts.maskPromptStrategy = maskPromptStrategy;
    }
    if (!pointsSpec.empty()) {
        if (!parsePointsSpec(pointsSpec, &nonInteractivePrompts.points, &nonInteractivePrompts.pointLabels)) {
            std::cerr << "[ERROR] Could not parse --points.\n";
            return 1;
        }
    } else if (!boxSpec.empty()) {
        SAM3Rect rect;
        if (!parseBoxSpec(boxSpec, &rect)) {
            std::cerr << "[ERROR] Could not parse --box.\n";
            return 1;
        }
        nonInteractivePrompts.rects.push_back(rect);
    }

    if (!pointsSpec.empty() || !boxSpec.empty() || !maskPath.empty()) {
        const auto decoderStart = std::chrono::high_resolution_clock::now();
        const Image<float> mask = sam.inferSingleFrame(
            SAM3Size(image.cols, image.rows),
            nonInteractivePrompts);
        const double decoderElapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - decoderStart).count();
        std::cout << "[INFO] Image decoder => " << decoderElapsedMs << " ms\n";
        const cv::Mat overlay = overlayMask(
            image,
            CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0));
        if (!saveOverlayPath.empty()) {
            cv::imwrite(saveOverlayPath, overlay);
            std::cout << "[INFO] Saved overlay to " << saveOverlayPath << '\n';
        }
        if (!noGui) {
            cv::imshow("SAM3 Image", overlay);
            while ((cv::waitKey(20) & 0xFF) != 27) {
            }
            cv::destroyAllWindows();
        }
        return 0;
    }

    if (noGui) {
        std::cerr << "[ERROR] --no_gui requires --points, --box, or --mask.\n";
        return 1;
    }

    AppState state;
    state.sam = &sam;
    state.original = image;
    state.originalSize = SAM3Size(image.cols, image.rows);
    state.mode = promptMode;
    state.display = image.clone();

    cv::namedWindow("SAM3 Image", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("SAM3 Image", onMouse, &state);
    cv::imshow("SAM3 Image", state.display);

    std::cout << "[INFO] Interactive mode ready. ESC to quit.\n";
    if (promptMode == PromptMode::SeedPoints) {
        std::cout << "[INFO] L-click = FG, R-click = BG, M-click = clear\n";
    } else {
        std::cout << "[INFO] L-drag = box, R/M-click = clear\n";
    }

    while ((cv::waitKey(20) & 0xFF) != 27) {
    }
    cv::destroyAllWindows();
    return 0;
}
