#include "SAM3.h"
#include "CVHelpers.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>

static std::wstring strToWstr(const std::string& value)
{
    const int sizeNeeded = MultiByteToWideChar(
        CP_UTF8,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        nullptr,
        0);
    std::wstring wideValue(sizeNeeded, 0);
    MultiByteToWideChar(
        CP_UTF8,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        wideValue.data(),
        sizeNeeded);
    return wideValue;
}
#endif

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace {

struct LoadedNpyArray {
    std::string descr;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

std::string lowerAsciiCopy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string trimCopy(const std::string& value)
{
    const auto first = std::find_if_not(
        value.begin(),
        value.end(),
        [](unsigned char ch) { return std::isspace(ch) != 0; });
    const auto last = std::find_if_not(
        value.rbegin(),
        value.rend(),
        [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
    if (first >= last) {
        return std::string();
    }
    return std::string(first, last);
}

uint16_t readUInt16LE(const std::vector<uint8_t>& bytes, size_t offset)
{
    return static_cast<uint16_t>(bytes[offset])
        | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t readUInt32LE(const std::vector<uint8_t>& bytes, size_t offset)
{
    return static_cast<uint32_t>(bytes[offset])
        | (static_cast<uint32_t>(bytes[offset + 1]) << 8)
        | (static_cast<uint32_t>(bytes[offset + 2]) << 16)
        | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

uint64_t readUInt64LE(const std::vector<uint8_t>& bytes, size_t offset)
{
    return static_cast<uint64_t>(bytes[offset])
        | (static_cast<uint64_t>(bytes[offset + 1]) << 8)
        | (static_cast<uint64_t>(bytes[offset + 2]) << 16)
        | (static_cast<uint64_t>(bytes[offset + 3]) << 24)
        | (static_cast<uint64_t>(bytes[offset + 4]) << 32)
        | (static_cast<uint64_t>(bytes[offset + 5]) << 40)
        | (static_cast<uint64_t>(bytes[offset + 6]) << 48)
        | (static_cast<uint64_t>(bytes[offset + 7]) << 56);
}

std::vector<uint8_t> readBinaryFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open file: " + path);
    }

    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error("Could not determine file size: " + path);
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0) {
        input.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    return bytes;
}

std::string parseHeaderStringValue(const std::string& header, const std::string& key)
{
    const std::string quotedKey = "'" + key + "'";
    const size_t keyPos = header.find(quotedKey);
    if (keyPos == std::string::npos) {
        return std::string();
    }

    const size_t colonPos = header.find(':', keyPos);
    const size_t firstQuote = header.find('\'', colonPos + 1);
    const size_t secondQuote = header.find('\'', firstQuote + 1);
    if (colonPos == std::string::npos || firstQuote == std::string::npos || secondQuote == std::string::npos) {
        return std::string();
    }
    return header.substr(firstQuote + 1, secondQuote - firstQuote - 1);
}

bool parseHeaderBoolValue(const std::string& header, const std::string& key)
{
    const std::string quotedKey = "'" + key + "'";
    const size_t keyPos = header.find(quotedKey);
    if (keyPos == std::string::npos) {
        return false;
    }

    const size_t colonPos = header.find(':', keyPos);
    if (colonPos == std::string::npos) {
        return false;
    }
    const std::string suffix = trimCopy(header.substr(colonPos + 1));
    return suffix.rfind("True", 0) == 0;
}

std::vector<int64_t> parseHeaderShape(const std::string& header)
{
    const size_t shapeKeyPos = header.find("'shape'");
    if (shapeKeyPos == std::string::npos) {
        return {};
    }

    const size_t openPos = header.find('(', shapeKeyPos);
    const size_t closePos = header.find(')', openPos + 1);
    if (openPos == std::string::npos || closePos == std::string::npos) {
        return {};
    }

    const std::string shapeText = header.substr(openPos + 1, closePos - openPos - 1);
    std::vector<int64_t> shape;
    std::stringstream stream(shapeText);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const std::string trimmed = trimCopy(item);
        if (trimmed.empty()) {
            continue;
        }
        shape.push_back(std::stoll(trimmed));
    }
    return shape;
}

size_t elementSizeFromDescr(const std::string& descr)
{
    size_t digitStart = descr.size();
    while (digitStart > 0 && std::isdigit(static_cast<unsigned char>(descr[digitStart - 1])) != 0) {
        --digitStart;
    }
    if (digitStart >= descr.size()) {
        throw std::runtime_error("Could not determine NPY element size from descr: " + descr);
    }
    return static_cast<size_t>(std::stoull(descr.substr(digitStart)));
}

size_t elementCountFromShape(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("NPY shape has a negative dimension.");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

void reorderFortranBytesToCOrder(const std::vector<int64_t>& shape,
                                 size_t elementSize,
                                 std::vector<uint8_t>* bytes)
{
    if (!bytes || shape.size() <= 1) {
        return;
    }

    const size_t elementCount = elementCountFromShape(shape);
    if (elementCount == 0) {
        bytes->clear();
        return;
    }
    if (elementSize == 0 || bytes->size() != elementCount * elementSize) {
        throw std::runtime_error("NPY payload size does not match shape and element size.");
    }

    std::vector<uint8_t> reordered(bytes->size());
    std::vector<size_t> coords(shape.size(), 0);
    for (size_t cIndex = 0; cIndex < elementCount; ++cIndex) {
        size_t remaining = cIndex;
        for (size_t dim = shape.size(); dim-- > 0;) {
            const size_t extent = static_cast<size_t>(shape[dim]);
            coords[dim] = remaining % extent;
            remaining /= extent;
        }

        size_t fortranIndex = 0;
        size_t fortranStride = 1;
        for (size_t dim = 0; dim < shape.size(); ++dim) {
            fortranIndex += coords[dim] * fortranStride;
            fortranStride *= static_cast<size_t>(shape[dim]);
        }

        std::memcpy(
            reordered.data() + cIndex * elementSize,
            bytes->data() + fortranIndex * elementSize,
            elementSize);
    }

    *bytes = std::move(reordered);
}

LoadedNpyArray parseNpyArray(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() < 10) {
        throw std::runtime_error("NPY payload is too small.");
    }

    constexpr std::array<uint8_t, 6> magic = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (!std::equal(magic.begin(), magic.end(), bytes.begin())) {
        throw std::runtime_error("NPY payload does not start with the NumPy magic header.");
    }

    const uint8_t major = bytes[6];
    const size_t headerLenOffset = 8;
    size_t headerLength = 0;
    size_t headerOffset = 0;

    if (major == 1) {
        headerLength = readUInt16LE(bytes, headerLenOffset);
        headerOffset = 10;
    } else if (major == 2) {
        headerLength = readUInt32LE(bytes, headerLenOffset);
        headerOffset = 12;
    } else {
        throw std::runtime_error("Unsupported NPY major version: " + std::to_string(major));
    }

    if (headerOffset + headerLength > bytes.size()) {
        throw std::runtime_error("NPY header exceeds the payload length.");
    }

    const std::string header(
        reinterpret_cast<const char*>(bytes.data() + headerOffset),
        headerLength);
    const bool fortranOrder = parseHeaderBoolValue(header, "fortran_order");

    LoadedNpyArray array;
    array.descr = parseHeaderStringValue(header, "descr");
    array.shape = parseHeaderShape(header);

    const size_t dataOffset = headerOffset + headerLength;
    array.bytes.assign(bytes.begin() + static_cast<ptrdiff_t>(dataOffset), bytes.end());
    if (fortranOrder) {
        reorderFortranBytesToCOrder(array.shape, elementSizeFromDescr(array.descr), &array.bytes);
    }
    return array;
}

void applyZip64Sizes(const std::vector<uint8_t>& bytes,
                     size_t extraFieldOffset,
                     size_t extraFieldLength,
                     bool needsUncompressedSize,
                     bool needsCompressedSize,
                     uint64_t* uncompressedSize,
                     uint64_t* compressedSize)
{
    if (!uncompressedSize || !compressedSize) {
        throw std::runtime_error("Invalid ZIP64 size output pointers.");
    }
    if (extraFieldOffset > bytes.size() || extraFieldLength > bytes.size() - extraFieldOffset) {
        throw std::runtime_error("Truncated ZIP extra field.");
    }

    const size_t extraEnd = extraFieldOffset + extraFieldLength;
    size_t cursor = extraFieldOffset;
    while (cursor + 4 <= extraEnd) {
        const uint16_t headerId = readUInt16LE(bytes, cursor);
        const uint16_t dataSize = readUInt16LE(bytes, cursor + 2);
        cursor += 4;
        if (dataSize > extraEnd - cursor) {
            throw std::runtime_error("Truncated ZIP extra field payload.");
        }

        if (headerId == 0x0001u) {
            size_t valueOffset = cursor;
            if (needsUncompressedSize) {
                if (valueOffset + 8 > cursor + dataSize) {
                    throw std::runtime_error("ZIP64 extra field is missing uncompressed size.");
                }
                *uncompressedSize = readUInt64LE(bytes, valueOffset);
                valueOffset += 8;
            }
            if (needsCompressedSize) {
                if (valueOffset + 8 > cursor + dataSize) {
                    throw std::runtime_error("ZIP64 extra field is missing compressed size.");
                }
                *compressedSize = readUInt64LE(bytes, valueOffset);
            }
            return;
        }

        cursor += dataSize;
    }

    if (needsUncompressedSize || needsCompressedSize) {
        throw std::runtime_error("ZIP64 extra field is required but was not found.");
    }
}

std::map<std::string, LoadedNpyArray> loadNpzArchive(const std::string& path)
{
    const auto archiveBytes = readBinaryFile(path);
    std::map<std::string, LoadedNpyArray> arrays;

    size_t offset = 0;
    while (offset + 4 <= archiveBytes.size()) {
        const uint32_t signature = readUInt32LE(archiveBytes, offset);
        if (signature == 0x06054b50u || signature == 0x02014b50u) {
            break;
        }
        if (signature != 0x04034b50u) {
            throw std::runtime_error("Unexpected ZIP local file header while reading: " + path);
        }

        if (offset + 30 > archiveBytes.size()) {
            throw std::runtime_error("Truncated ZIP local file header in: " + path);
        }

        const uint16_t compressionMethod = readUInt16LE(archiveBytes, offset + 8);
        uint64_t compressedSize = readUInt32LE(archiveBytes, offset + 18);
        uint64_t uncompressedSize = readUInt32LE(archiveBytes, offset + 22);
        const uint16_t fileNameLength = readUInt16LE(archiveBytes, offset + 26);
        const uint16_t extraFieldLength = readUInt16LE(archiveBytes, offset + 28);
        const size_t fileNameOffset = offset + 30;
        const size_t extraFieldOffset = fileNameOffset + fileNameLength;
        const size_t dataOffset = extraFieldOffset + extraFieldLength;

        if (dataOffset > archiveBytes.size()) {
            throw std::runtime_error("Truncated ZIP entry header in: " + path);
        }

        const bool needsCompressedSize = compressedSize == 0xFFFFFFFFull;
        const bool needsUncompressedSize = uncompressedSize == 0xFFFFFFFFull;
        if (needsCompressedSize || needsUncompressedSize) {
            applyZip64Sizes(
                archiveBytes,
                extraFieldOffset,
                extraFieldLength,
                needsUncompressedSize,
                needsCompressedSize,
                &uncompressedSize,
                &compressedSize);
        }

        if (compressedSize > archiveBytes.size() - dataOffset) {
            throw std::runtime_error("Truncated ZIP entry payload in: " + path);
        }

        const std::string fileName(
            reinterpret_cast<const char*>(archiveBytes.data() + fileNameOffset),
            fileNameLength);
        if (compressionMethod != 0) {
            throw std::runtime_error("Compressed NPZ entries are not supported: " + fileName);
        }
        if (compressedSize != uncompressedSize) {
            throw std::runtime_error("Unexpected ZIP stored entry size mismatch: " + fileName);
        }

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".npy") {
            const size_t payloadSize = static_cast<size_t>(compressedSize);
            std::vector<uint8_t> npyBytes(
                archiveBytes.begin() + static_cast<ptrdiff_t>(dataOffset),
                archiveBytes.begin() + static_cast<ptrdiff_t>(dataOffset + payloadSize));
            arrays[fileName.substr(0, fileName.size() - 4)] = parseNpyArray(npyBytes);
        }

        offset = dataOffset + compressedSize;
    }

    return arrays;
}

std::vector<SAM3Node> getSessionNodesInternal(Ort::Session* session, bool isInput)
{
    std::vector<SAM3Node> nodes;
    if (!session) {
        return nodes;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = isInput ? session->GetInputCount() : session->GetOutputCount();
    nodes.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        SAM3Node node;
        auto name = isInput ? session->GetInputNameAllocated(i, allocator)
                            : session->GetOutputNameAllocated(i, allocator);
        node.name = name.get();
        auto shape = isInput ? session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()
                             : session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        node.dim.assign(shape.begin(), shape.end());
        nodes.push_back(std::move(node));
    }

    return nodes;
}

void ortThrowIf(OrtStatus* status, const char* what)
{
    if (!status) {
        return;
    }

    const char* message = Ort::GetApi().GetErrorMessage(status);
    std::ostringstream stream;
    stream << what << " : " << (message ? message : "(null)");
    Ort::GetApi().ReleaseStatus(status);
    throw std::runtime_error(stream.str());
}

bool envBool(const char* name, bool fallback)
{
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }

    const std::string lowered = lowerAsciiCopy(value);
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    return fallback;
}

int envInt(const char* name, int fallback, int minValue = 1)
{
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }

    try {
        return std::max(minValue, std::stoi(value));
    } catch (...) {
        return fallback;
    }
}

GraphOptimizationLevel resolveGraphOptimizationLevel(const std::string& device,
                                                     GraphOptimizationLevel fallback)
{
    const char* value = std::getenv("SAM3_ORT_GRAPH_OPT");
    if (!value || !*value) {
        return fallback;
    }

    const std::string lowered = lowerAsciiCopy(value);
    if (lowered == "disable" || lowered == "disabled" || lowered == "none" || lowered == "off") {
        return GraphOptimizationLevel::ORT_DISABLE_ALL;
    }
    if (lowered == "basic") {
        return GraphOptimizationLevel::ORT_ENABLE_BASIC;
    }
    if (lowered == "extended") {
        return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    }
    if (lowered == "all" || lowered == "full" || lowered == "aggressive") {
        return GraphOptimizationLevel::ORT_ENABLE_ALL;
    }
    return device == "cpu" ? GraphOptimizationLevel::ORT_ENABLE_ALL : fallback;
}

std::vector<int64_t> concreteEncoderInputShape(const std::vector<int64_t>& modelShape)
{
    std::vector<int64_t> concreteShape = modelShape;
    for (size_t index = 0; index < concreteShape.size(); ++index) {
        if (concreteShape[index] > 0) {
            continue;
        }

        if (index == 0) {
            concreteShape[index] = 1;
        } else if (index == 1) {
            concreteShape[index] = 3;
        }
    }
    return concreteShape;
}

#if defined(_WIN32)
std::wstring trimWideCopy(const std::wstring& value)
{
    const auto first = std::find_if_not(
        value.begin(),
        value.end(),
        [](wchar_t ch) { return std::iswspace(ch) != 0; });
    const auto last = std::find_if_not(
        value.rbegin(),
        value.rend(),
        [](wchar_t ch) { return std::iswspace(ch) != 0; }).base();
    if (first >= last) {
        return std::wstring();
    }
    return std::wstring(first, last);
}

std::wstring getEnvWide(const wchar_t* name)
{
    const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    if (size == 0) {
        return std::wstring();
    }

    std::wstring value(size - 1, L'\0');
    GetEnvironmentVariableW(name, value.data(), size);
    return value;
}

std::vector<std::wstring> splitWidePathList(const std::wstring& value)
{
    std::vector<std::wstring> parts;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t separator = value.find(L';', start);
        std::wstring part = separator == std::wstring::npos
            ? value.substr(start)
            : value.substr(start, separator - start);
        part = trimWideCopy(part);
        if (part.size() >= 2 && part.front() == L'"' && part.back() == L'"') {
            part = part.substr(1, part.size() - 2);
        }
        if (!part.empty()) {
            parts.push_back(std::move(part));
        }
        if (separator == std::wstring::npos) {
            break;
        }
        start = separator + 1;
    }
    return parts;
}

std::wstring getExecutableDirectory()
{
    std::wstring buffer(MAX_PATH, L'\0');
    while (true) {
        const DWORD written = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (written == 0) {
            return std::wstring();
        }
        if (written < buffer.size() - 1) {
            buffer.resize(written);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }

    std::filesystem::path exePath(buffer);
    return exePath.parent_path().native();
}

void appendExistingDirectory(const std::filesystem::path& candidate,
                             std::vector<std::wstring>* directories,
                             std::set<std::wstring>* seen)
{
    if (!directories || !seen || candidate.empty()) {
        return;
    }

    std::error_code error;
    if (!std::filesystem::exists(candidate, error) || !std::filesystem::is_directory(candidate, error)) {
        return;
    }

    const std::wstring normalized = candidate.lexically_normal().native();
    if (seen->insert(normalized).second) {
        directories->push_back(normalized);
    }
}

std::vector<std::wstring> collectCudaSearchDirectories()
{
    std::vector<std::wstring> directories;
    std::set<std::wstring> seen;

    appendExistingDirectory(getExecutableDirectory(), &directories, &seen);

    std::error_code error;
    appendExistingDirectory(std::filesystem::current_path(error), &directories, &seen);

    for (const auto& entry : splitWidePathList(getEnvWide(L"SAM3_ORT_DLL_DIRS"))) {
        appendExistingDirectory(entry, &directories, &seen);
    }
    for (const auto& entry : splitWidePathList(getEnvWide(L"SAM3_CUDA_DLL_DIRS"))) {
        appendExistingDirectory(entry, &directories, &seen);
    }
    for (const auto& entry : splitWidePathList(getEnvWide(L"PATH"))) {
        appendExistingDirectory(entry, &directories, &seen);
    }

    const std::filesystem::path programFiles = getEnvWide(L"ProgramFiles");
    if (!programFiles.empty()) {
        const std::filesystem::path cudaRoot = programFiles / "NVIDIA GPU Computing Toolkit" / "CUDA";
        if (std::filesystem::exists(cudaRoot, error)) {
            std::vector<std::filesystem::path> cudaBins;
            for (const auto& entry : std::filesystem::directory_iterator(cudaRoot, error)) {
                if (!entry.is_directory(error)) {
                    continue;
                }
                const std::filesystem::path binDir = entry.path() / "bin";
                if (std::filesystem::exists(binDir, error) && std::filesystem::is_directory(binDir, error)) {
                    cudaBins.push_back(binDir);
                }
            }
            std::sort(cudaBins.begin(), cudaBins.end(), [](const auto& a, const auto& b) {
                return a.native() > b.native();
            });
            for (const auto& binDir : cudaBins) {
                appendExistingDirectory(binDir, &directories, &seen);
            }
        }

        const std::filesystem::path cudnnRoot = programFiles / "NVIDIA" / "CUDNN";
        if (std::filesystem::exists(cudnnRoot, error)) {
            std::vector<std::filesystem::path> cudnnBins;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(cudnnRoot, error)) {
                if (!entry.is_regular_file(error)) {
                    continue;
                }
                const std::wstring filename = entry.path().filename().native();
                if (filename.rfind(L"cudnn", 0) == 0 && entry.path().extension() == ".dll") {
                    cudnnBins.push_back(entry.path().parent_path());
                }
            }
            std::sort(cudnnBins.begin(), cudnnBins.end(), [](const auto& a, const auto& b) {
                return a.native() > b.native();
            });
            for (const auto& binDir : cudnnBins) {
                appendExistingDirectory(binDir, &directories, &seen);
            }
        }
    }

    return directories;
}

HMODULE loadDllFromSearchDirectories(const std::wstring& dllName,
                                     const std::vector<std::wstring>& directories)
{
    if (dllName.empty()) {
        return nullptr;
    }

    if (HMODULE existing = GetModuleHandleW(dllName.c_str())) {
        return existing;
    }

    if (HMODULE loaded = LoadLibraryW(dllName.c_str())) {
        return loaded;
    }

    for (const auto& directory : directories) {
        const std::filesystem::path candidate = std::filesystem::path(directory) / dllName;
        std::error_code error;
        if (!std::filesystem::exists(candidate, error)) {
            continue;
        }

        if (HMODULE loaded = LoadLibraryExW(candidate.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) {
            return loaded;
        }
    }

    return nullptr;
}

bool preloadCudaWindowsRuntime()
{
    static int cached = -1;
    static std::vector<HMODULE> loadedModules;
    if (cached != -1) {
        return cached != 0;
    }

    const std::vector<std::wstring> searchDirectories = collectCudaSearchDirectories();
    const std::array<std::wstring, 11> requiredDlls = {
        L"cudart64_12.dll",
        L"cublas64_12.dll",
        L"cublasLt64_12.dll",
        L"cudnn64_9.dll",
        L"cudnn_adv64_9.dll",
        L"cudnn_cnn64_9.dll",
        L"cudnn_engines_precompiled64_9.dll",
        L"cudnn_engines_runtime_compiled64_9.dll",
        L"cudnn_graph64_9.dll",
        L"cudnn_heuristic64_9.dll",
        L"cudnn_ops64_9.dll",
    };
    const std::array<std::wstring, 6> optionalDlls = {
        L"cufft64_11.dll",
        L"cufftw64_11.dll",
        L"curand64_10.dll",
        L"cusparse64_12.dll",
        L"nvrtc64_120_0.dll",
        L"nvrtc-builtins64_125.dll",
    };

    for (const auto& dllName : requiredDlls) {
        HMODULE loaded = loadDllFromSearchDirectories(dllName, searchDirectories);
        if (!loaded) {
            cached = 0;
            return false;
        }
        loadedModules.push_back(loaded);
    }

    for (const auto& dllName : optionalDlls) {
        if (HMODULE loaded = loadDllFromSearchDirectories(dllName, searchDirectories)) {
            loadedModules.push_back(loaded);
        }
    }

    cached = 1;
    return true;
}
#endif

} // namespace

SAM3::SAM3() = default;

SAM3::~SAM3()
{
    clearSessions();
}

std::string SAM3::lowerCopy(const std::string& value)
{
    std::string lowered = value;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lowered;
}

bool SAM3::modelExists(const std::string& modelPath) const
{
    std::ifstream file(modelPath.c_str(), std::ios::binary);
    return file.good();
}

bool SAM3::clearSessions()
{
    try {
        m_encoderSession.reset();
        m_imageDecoderSession.reset();
        m_trackerDecoderSession.reset();
        m_memoryAttentionSession.reset();
        m_memoryEncoderSession.reset();

        m_encoderInputNodes.clear();
        m_encoderOutputNodes.clear();
        m_imageDecoderInputNodes.clear();
        m_imageDecoderOutputNodes.clear();
        m_trackerDecoderInputNodes.clear();
        m_trackerDecoderOutputNodes.clear();
        m_memoryAttentionInputNodes.clear();
        m_memoryAttentionOutputNodes.clear();
        m_memoryEncoderInputNodes.clear();
        m_memoryEncoderOutputNodes.clear();

        m_encoderInputNames.clear();
        m_encoderOutputNames.clear();
        m_imageDecoderInputNames.clear();
        m_imageDecoderOutputNames.clear();
        m_trackerDecoderInputNames.clear();
        m_trackerDecoderOutputNames.clear();
        m_memoryAttentionInputNames.clear();
        m_memoryAttentionOutputNames.clear();
        m_memoryEncoderInputNames.clear();
        m_memoryEncoderOutputNames.clear();

        m_inputShapeEncoder.clear();

        m_encoderImageEmb0Index = -1;
        m_encoderImageEmb1Index = -1;
        m_encoderImageEmb2Index = -1;
        m_imageDecoderPredMasksIndex = -1;
        m_imageDecoderIouScoresIndex = -1;
        m_trackerDecoderObjPtrIndex = -1;
        m_trackerDecoderPredMaskHighResIndex = -1;
        m_trackerDecoderObjectScoreIndex = -1;
        m_trackerDecoderIouScoresIndex = -1;
        m_memoryAttentionFusedFeatIndex = -1;
        m_memoryEncoderFeaturesIndex = -1;
        m_memoryEncoderPosIndex = -1;
        m_staticNumMemFrames = 0;
        m_staticNumObjPtrs = 0;

        m_cachedEncoderOutputs.clear();
        m_cachedEncoderHostCopy = CachedEncoderOutputs();
        m_hasCachedEncoderHostCopy = false;

        m_constants = SAM3Constants();
        m_hasVideoConstants = false;

        resetMemory();
        m_device = "cpu";
    } catch (...) {
        return false;
    }

    return true;
}

void SAM3::resetMemory()
{
    m_hasConditioningState = false;
    m_conditioningState = TrackerFrameState();
    m_nonConditioningStates.clear();
    m_segmentFrameIndex = 0;
    m_memoryObjPtrsScratch.clear();
    m_memoryObjTposScratch.clear();
    m_memoryMaskFeatsScratch.clear();
    m_memoryMaskPosScratch.clear();
    m_memoryMaskTposScratch.clear();
}

SAM3Size SAM3::getInputSize() const
{
    if (m_inputShapeEncoder.size() >= 4) {
        return SAM3Size(
            static_cast<int>(m_inputShapeEncoder[3]),
            static_cast<int>(m_inputShapeEncoder[2]));
    }
    return SAM3Size();
}

void SAM3::setupSessionOptions(Ort::SessionOptions& options,
                               int threadsNumber,
                               GraphOptimizationLevel optLevel,
                               const std::string& device)
{
    const int safeThreads = std::max(1, threadsNumber);
    const int safeInterOpThreads = envInt("SAM3_ORT_INTER_OP_THREADS", 1, 1);
    const bool enableCpuArena = envBool("SAM3_ORT_CPU_ARENA", true);
    const bool enableMemPattern = envBool("SAM3_ORT_MEM_PATTERN", true);
    options.SetIntraOpNumThreads(safeThreads);
    options.SetInterOpNumThreads(safeInterOpThreads);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(resolveGraphOptimizationLevel(device, optLevel));
    if (enableCpuArena) {
        options.EnableCpuMemArena();
    } else {
        options.DisableCpuMemArena();
    }
    if (enableMemPattern) {
        options.EnableMemPattern();
    } else {
        options.DisableMemPattern();
    }

    if (device.rfind("cuda:", 0) == 0) {
#if !defined(__APPLE__)
#if defined(_WIN32)
        if (!preloadCudaWindowsRuntime()) {
            std::cerr
                << "[WARN] Could not preload the CUDA/cuDNN runtime from PATH or the default install locations. "
                << "If CUDA session initialization fails, add the CUDA/cuDNN bin folders to PATH or SAM3_CUDA_DLL_DIRS.\n";
        }
#endif
        OrtCUDAProviderOptions cudaOptions{};
        try {
            cudaOptions.device_id = std::stoi(device.substr(5));
        } catch (...) {
            cudaOptions.device_id = 0;
        }
        options.AppendExecutionProvider_CUDA(cudaOptions);
#endif
        ortThrowIf(
            OrtSessionOptionsAppendExecutionProvider_CPU(options, enableCpuArena ? 1 : 0),
            "Append CPU EP (fallback) failed");
        return;
    }

    if (device.rfind("coreml", 0) == 0) {
#ifdef __APPLE__
        ortThrowIf(
            OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0),
            "Append CoreML EP failed");
        return;
#endif
        ortThrowIf(
            OrtSessionOptionsAppendExecutionProvider_CPU(options, enableCpuArena ? 1 : 0),
            "Append CPU EP (fallback) failed");
        return;
    }

    ortThrowIf(
        OrtSessionOptionsAppendExecutionProvider_CPU(options, enableCpuArena ? 1 : 0),
        "Append CPU EP failed");
}

std::vector<SAM3Node> SAM3::getSessionNodes(Ort::Session* session, bool isInput)
{
    return getSessionNodesInternal(session, isInput);
}

std::vector<const char*> SAM3::collectNodeNames(const std::vector<SAM3Node>& nodes)
{
    std::vector<const char*> names;
    names.reserve(nodes.size());
    for (const auto& node : nodes) {
        names.push_back(node.name.c_str());
    }
    return names;
}

int SAM3::findNodeIndex(const std::vector<SAM3Node>& nodes, const std::string& key)
{
    const std::string loweredKey = lowerCopy(key);
    for (size_t index = 0; index < nodes.size(); ++index) {
        if (lowerCopy(nodes[index].name).find(loweredKey) != std::string::npos) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int SAM3::findNameIndex(const std::vector<const char*>& names, const std::string& key)
{
    const std::string loweredKey = lowerCopy(key);
    for (size_t index = 0; index < names.size(); ++index) {
        const char* name = names[index];
        if (name && lowerCopy(name).find(loweredKey) != std::string::npos) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

bool SAM3::initializeNamedSession(std::unique_ptr<Ort::Session>* sessionOut,
                                  const Ort::Env& env,
                                  const std::string& modelPath,
                                  const Ort::SessionOptions& options,
                                  std::vector<SAM3Node>* inputNodes,
                                  std::vector<SAM3Node>* outputNodes,
                                  std::vector<const char*>* inputNames,
                                  std::vector<const char*>* outputNames)
{
    try {
        std::unique_ptr<Ort::Session> session;
#ifdef _WIN32
        const std::wstring widePath = strToWstr(modelPath);
        session = std::make_unique<Ort::Session>(env, widePath.c_str(), options);
#else
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
#endif

        if (inputNodes) {
            *inputNodes = getSessionNodesInternal(session.get(), true);
        }
        if (outputNodes) {
            *outputNodes = getSessionNodesInternal(session.get(), false);
        }
        if (inputNames && inputNodes) {
            *inputNames = collectNodeNames(*inputNodes);
        }
        if (outputNames && outputNodes) {
            *outputNames = collectNodeNames(*outputNodes);
        }

        *sessionOut = std::move(session);
        return true;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] Failed to load session " << modelPath << " => " << error.what() << '\n';
        return false;
    }
}

std::variant<std::vector<Ort::Value>, std::string> SAM3::runSession(
    Ort::Session* session,
    const std::vector<const char*>& inputNames,
    const std::vector<const char*>& outputNames,
    const std::vector<Ort::Value>& inputTensors,
    const std::string& debugName)
{
    if (!session) {
        return std::string("[ERROR] runSession(" + debugName + "): session is null.");
    }

    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            const_cast<Ort::Value*>(inputTensors.data()),
            inputTensors.size(),
            outputNames.data(),
            outputNames.size());
        return outputs;
    } catch (const std::exception& error) {
        return std::string("[ERROR] runSession(" + debugName + ") => " + error.what());
    }
}

bool SAM3::initializeImage(const std::string& encoderPath,
                           const std::string& decoderPath,
                           int threadsNumber,
                           const std::string& device)
{
    clearSessions();
    m_device = device;

    if (!modelExists(encoderPath) || !modelExists(decoderPath)) {
        std::cerr << "[ERROR] Model file not found.\n";
        return false;
    }

    Ort::SessionOptions encoderOptions;
    Ort::SessionOptions decoderOptions;
    const GraphOptimizationLevel optLevel =
        device == "cpu" ? GraphOptimizationLevel::ORT_ENABLE_ALL : GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    setupSessionOptions(
        encoderOptions,
        threadsNumber,
        optLevel,
        device);
    setupSessionOptions(
        decoderOptions,
        threadsNumber,
        optLevel,
        device);

    if (!initializeNamedSession(
            &m_encoderSession,
            m_encoderEnv,
            encoderPath,
            encoderOptions,
            &m_encoderInputNodes,
            &m_encoderOutputNodes,
            &m_encoderInputNames,
            &m_encoderOutputNames)) {
        return false;
    }

    if (!initializeNamedSession(
            &m_imageDecoderSession,
            m_imageDecoderEnv,
            decoderPath,
            decoderOptions,
            &m_imageDecoderInputNodes,
            &m_imageDecoderOutputNodes,
            &m_imageDecoderInputNames,
            &m_imageDecoderOutputNames)) {
        return false;
    }

    if (m_encoderInputNodes.empty()) {
        std::cerr << "[ERROR] Encoder did not expose its input metadata.\n";
        return false;
    }
    m_inputShapeEncoder = concreteEncoderInputShape(m_encoderInputNodes.front().dim);
    if (m_inputShapeEncoder.size() < 4) {
        std::cerr << "[ERROR] Could not determine the encoder input shape.\n";
        return false;
    }
    if (m_inputShapeEncoder[2] <= 0 || m_inputShapeEncoder[3] <= 0) {
        std::cerr << "[ERROR] Encoder input metadata has unresolved spatial dims.\n";
        return false;
    }

    m_encoderImageEmb0Index = findNodeIndex(m_encoderOutputNodes, "image_embeddings.0");
    m_encoderImageEmb1Index = findNodeIndex(m_encoderOutputNodes, "image_embeddings.1");
    m_encoderImageEmb2Index = findNodeIndex(m_encoderOutputNodes, "image_embeddings.2");
    if (m_encoderImageEmb0Index < 0 || m_encoderImageEmb1Index < 0 || m_encoderImageEmb2Index < 0) {
        std::cerr << "[ERROR] Encoder outputs are missing one or more image_embeddings.* tensors.\n";
        return false;
    }

    m_imageDecoderPredMasksIndex = findNodeIndex(m_imageDecoderOutputNodes, "pred_masks");
    m_imageDecoderIouScoresIndex = findNodeIndex(m_imageDecoderOutputNodes, "iou_scores");
    if (m_imageDecoderPredMasksIndex < 0 || m_imageDecoderIouScoresIndex < 0) {
        std::cerr << "[ERROR] Image decoder outputs are missing pred_masks or iou_scores.\n";
        return false;
    }

    return true;
}

bool SAM3::loadVideoConstants(const std::string& constantsPath)
{
    auto arrays = loadNpzArchive(constantsPath);

    auto loadFloatArray = [&](const std::string& key,
                              std::vector<float>* valuesOut,
                              std::vector<int64_t>* shapeOut) -> bool {
        const auto it = arrays.find(key);
        if (it == arrays.end() || it->second.descr != "<f4") {
            return false;
        }
        if (it->second.bytes.size() % sizeof(float) != 0) {
            return false;
        }

        valuesOut->resize(it->second.bytes.size() / sizeof(float));
        std::memcpy(valuesOut->data(), it->second.bytes.data(), it->second.bytes.size());
        *shapeOut = it->second.shape;
        return true;
    };

    auto loadIntScalar = [&](const std::string& key, int* valueOut) -> bool {
        const auto it = arrays.find(key);
        if (it == arrays.end() || it->second.descr != "<i8" || it->second.bytes.size() < sizeof(int64_t)) {
            return false;
        }
        int64_t value = 0;
        std::memcpy(&value, it->second.bytes.data(), sizeof(int64_t));
        *valueOut = static_cast<int>(value);
        return true;
    };

    auto loadBoolScalar = [&](const std::string& key, bool* valueOut) -> bool {
        int value = 0;
        if (!loadIntScalar(key, &value)) {
            return false;
        }
        *valueOut = value != 0;
        return true;
    };

    auto loadFloatScalar = [&](const std::string& key, float* valueOut) -> bool {
        const auto it = arrays.find(key);
        if (it == arrays.end() || it->second.descr != "<f4" || it->second.bytes.size() < sizeof(float)) {
            return false;
        }
        std::memcpy(valueOut, it->second.bytes.data(), sizeof(float));
        return true;
    };

    if (!loadFloatArray("no_mem_embed_bchw", &m_constants.noMemEmbed, &m_constants.noMemEmbedShape)
        || !loadFloatArray(
            "current_vision_pos_embed",
            &m_constants.currentVisionPosEmbed,
            &m_constants.currentVisionPosEmbedShape)) {
        std::cerr << "[ERROR] Failed to read required SAM3 tracker constants from " << constantsPath << '\n';
        return false;
    }

    loadIntScalar("num_maskmem", &m_constants.numMaskmem);
    loadIntScalar("max_obj_ptrs", &m_constants.maxObjPtrs);
    loadIntScalar("max_cond_frames_in_attn", &m_constants.maxCondFramesInAttn);
    loadBoolScalar("keep_first_cond_frame", &m_constants.keepFirstCondFrame);
    loadIntScalar("memory_temporal_stride_for_eval", &m_constants.memoryTemporalStrideForEval);
    loadBoolScalar("use_memory_selection", &m_constants.useMemorySelection);
    loadFloatScalar("mf_threshold", &m_constants.mfThreshold);
    loadIntScalar("export_max_mem_frames", &m_constants.exportMaxMemFrames);
    loadIntScalar("export_max_obj_ptrs", &m_constants.exportMaxObjPtrs);

    m_hasVideoConstants = true;
    return true;
}

bool SAM3::initializeVideo(const std::string& encoderPath,
                           const std::string& decoderPath,
                           const std::string& memoryAttentionPath,
                           const std::string& memoryEncoderPath,
                           const std::string& constantsPath,
                           int threadsNumber,
                           const std::string& device)
{
    clearSessions();
    m_device = device;

    if (!modelExists(encoderPath)
        || !modelExists(decoderPath)
        || !modelExists(memoryAttentionPath)
        || !modelExists(memoryEncoderPath)
        || !modelExists(constantsPath)) {
        std::cerr << "[ERROR] One or more video tracker artifacts are missing.\n";
        return false;
    }

    Ort::SessionOptions encoderOptions;
    Ort::SessionOptions decoderOptions;
    Ort::SessionOptions memoryAttentionOptions;
    Ort::SessionOptions memoryEncoderOptions;
    const GraphOptimizationLevel optLevel =
        device == "cpu" ? GraphOptimizationLevel::ORT_ENABLE_ALL : GraphOptimizationLevel::ORT_ENABLE_EXTENDED;

    setupSessionOptions(
        encoderOptions,
        threadsNumber,
        optLevel,
        device);
    setupSessionOptions(
        decoderOptions,
        threadsNumber,
        optLevel,
        device);
    setupSessionOptions(
        memoryAttentionOptions,
        threadsNumber,
        optLevel,
        device);
    setupSessionOptions(
        memoryEncoderOptions,
        threadsNumber,
        optLevel,
        device);

    if (!initializeNamedSession(
            &m_encoderSession,
            m_encoderEnv,
            encoderPath,
            encoderOptions,
            &m_encoderInputNodes,
            &m_encoderOutputNodes,
            &m_encoderInputNames,
            &m_encoderOutputNames)) {
        return false;
    }
    if (!initializeNamedSession(
            &m_trackerDecoderSession,
            m_trackerDecoderEnv,
            decoderPath,
            decoderOptions,
            &m_trackerDecoderInputNodes,
            &m_trackerDecoderOutputNodes,
            &m_trackerDecoderInputNames,
            &m_trackerDecoderOutputNames)) {
        return false;
    }
    if (!initializeNamedSession(
            &m_memoryAttentionSession,
            m_memoryAttentionEnv,
            memoryAttentionPath,
            memoryAttentionOptions,
            &m_memoryAttentionInputNodes,
            &m_memoryAttentionOutputNodes,
            &m_memoryAttentionInputNames,
            &m_memoryAttentionOutputNames)) {
        return false;
    }
    if (!initializeNamedSession(
            &m_memoryEncoderSession,
            m_memoryEncoderEnv,
            memoryEncoderPath,
            memoryEncoderOptions,
            &m_memoryEncoderInputNodes,
            &m_memoryEncoderOutputNodes,
            &m_memoryEncoderInputNames,
            &m_memoryEncoderOutputNames)) {
        return false;
    }
    if (!loadVideoConstants(constantsPath)) {
        return false;
    }

    if (m_encoderInputNodes.empty()) {
        std::cerr << "[ERROR] Encoder did not expose its input metadata.\n";
        return false;
    }
    m_inputShapeEncoder = concreteEncoderInputShape(m_encoderInputNodes.front().dim);
    if (m_inputShapeEncoder.size() < 4) {
        std::cerr << "[ERROR] Could not determine the encoder input shape.\n";
        return false;
    }
    if (m_inputShapeEncoder[2] <= 0 || m_inputShapeEncoder[3] <= 0) {
        std::cerr << "[ERROR] Encoder input metadata has unresolved spatial dims.\n";
        return false;
    }

    m_encoderImageEmb0Index = findNodeIndex(m_encoderOutputNodes, "image_embeddings.0");
    m_encoderImageEmb1Index = findNodeIndex(m_encoderOutputNodes, "image_embeddings.1");
    m_encoderImageEmb2Index = findNodeIndex(m_encoderOutputNodes, "image_embeddings.2");
    if (m_encoderImageEmb0Index < 0 || m_encoderImageEmb1Index < 0 || m_encoderImageEmb2Index < 0) {
        std::cerr << "[ERROR] Encoder outputs are missing one or more image_embeddings.* tensors.\n";
        return false;
    }

    m_trackerDecoderObjPtrIndex = findNodeIndex(m_trackerDecoderOutputNodes, "obj_ptr");
    m_trackerDecoderPredMaskHighResIndex = findNodeIndex(m_trackerDecoderOutputNodes, "pred_mask_high_res");
    m_trackerDecoderObjectScoreIndex = findNodeIndex(m_trackerDecoderOutputNodes, "object_score_logits");
    m_trackerDecoderIouScoresIndex = findNodeIndex(m_trackerDecoderOutputNodes, "iou_scores");
    if (m_trackerDecoderObjPtrIndex < 0
        || m_trackerDecoderPredMaskHighResIndex < 0
        || m_trackerDecoderObjectScoreIndex < 0) {
        std::cerr << "[ERROR] Tracker decoder outputs are missing obj_ptr, pred_mask_high_res, or object_score_logits.\n";
        return false;
    }

    m_memoryAttentionFusedFeatIndex = findNodeIndex(m_memoryAttentionOutputNodes, "fused_feat");
    if (m_memoryAttentionFusedFeatIndex < 0) {
        m_memoryAttentionFusedFeatIndex = 0;
    }
    m_memoryEncoderFeaturesIndex = findNodeIndex(m_memoryEncoderOutputNodes, "maskmem_features");
    m_memoryEncoderPosIndex = findNodeIndex(m_memoryEncoderOutputNodes, "maskmem_pos_enc");
    if (m_memoryEncoderFeaturesIndex < 0 || m_memoryEncoderPosIndex < 0) {
        std::cerr << "[ERROR] Memory encoder outputs are missing maskmem_features or maskmem_pos_enc.\n";
        return false;
    }

    const int memoryMaskIndex = findNodeIndex(m_memoryAttentionInputNodes, "memory_mask_feats");
    const int memoryObjIndex = findNodeIndex(m_memoryAttentionInputNodes, "memory_obj_ptrs");
    if (memoryMaskIndex >= 0 && memoryMaskIndex < static_cast<int>(m_memoryAttentionInputNodes.size())) {
        const auto& dims = m_memoryAttentionInputNodes[static_cast<size_t>(memoryMaskIndex)].dim;
        if (!dims.empty() && dims[0] > 0) {
            m_staticNumMemFrames = static_cast<int>(dims[0]);
        }
    }
    if (memoryObjIndex >= 0 && memoryObjIndex < static_cast<int>(m_memoryAttentionInputNodes.size())) {
        const auto& dims = m_memoryAttentionInputNodes[static_cast<size_t>(memoryObjIndex)].dim;
        if (!dims.empty() && dims[0] > 0) {
            m_staticNumObjPtrs = static_cast<int>(dims[0]);
        }
    }
    if (m_staticNumMemFrames <= 0) {
        m_staticNumMemFrames = std::max(1, m_constants.exportMaxMemFrames);
    }
    if (m_staticNumObjPtrs <= 0) {
        m_staticNumObjPtrs = std::max(1, m_constants.exportMaxObjPtrs);
    }

    return true;
}

bool SAM3::captureCachedEncoderOutputs(CachedEncoderOutputs* outputs) const
{
    if (!outputs) {
        return false;
    }

    const int requiredMaxIndex = std::max({m_encoderImageEmb0Index, m_encoderImageEmb1Index, m_encoderImageEmb2Index});
    if (requiredMaxIndex < 0 || m_cachedEncoderOutputs.size() <= static_cast<size_t>(requiredMaxIndex)) {
        return false;
    }

    extractTensorData(m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb0Index)], outputs->imageEmb0.values, outputs->imageEmb0.shape);
    extractTensorData(m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb1Index)], outputs->imageEmb1.values, outputs->imageEmb1.shape);
    extractTensorData(m_cachedEncoderOutputs[static_cast<size_t>(m_encoderImageEmb2Index)], outputs->imageEmb2.values, outputs->imageEmb2.shape);
    return !outputs->imageEmb0.values.empty()
        && !outputs->imageEmb1.values.empty()
        && !outputs->imageEmb2.values.empty();
}

bool SAM3::restoreCachedEncoderOutputs(const CachedEncoderOutputs& outputs)
{
    if (outputs.imageEmb0.values.empty()
        || outputs.imageEmb1.values.empty()
        || outputs.imageEmb2.values.empty()) {
        return false;
    }

    const int requiredMaxIndex = std::max({m_encoderImageEmb0Index, m_encoderImageEmb1Index, m_encoderImageEmb2Index});
    if (requiredMaxIndex < 0) {
        return false;
    }

    std::vector<Ort::Value> restored(static_cast<size_t>(requiredMaxIndex + 1));
    restored[static_cast<size_t>(m_encoderImageEmb0Index)] =
        createTensor<float>(m_memoryInfo, outputs.imageEmb0.values, outputs.imageEmb0.shape);
    restored[static_cast<size_t>(m_encoderImageEmb1Index)] =
        createTensor<float>(m_memoryInfo, outputs.imageEmb1.values, outputs.imageEmb1.shape);
    restored[static_cast<size_t>(m_encoderImageEmb2Index)] =
        createTensor<float>(m_memoryInfo, outputs.imageEmb2.values, outputs.imageEmb2.shape);

    m_cachedEncoderOutputs = std::move(restored);
    m_cachedEncoderHostCopy = outputs;
    m_hasCachedEncoderHostCopy = true;
    return true;
}

bool SAM3::preprocessImage(const Image<float>& originalImage)
{
    try {
        const SAM3Size targetSize = getInputSize();
        const std::vector<float> encoderData = CVHelpers::resizeImageToPlanarTensor(
            originalImage,
            targetSize.width,
            targetSize.height);
        Ort::Value inputTensor = createTensor<float>(m_memoryInfo, encoderData, m_inputShapeEncoder);
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(inputTensor));

        auto result = runSession(
            m_encoderSession.get(),
            m_encoderInputNames,
            m_encoderOutputNames,
            inputs,
            "encoder");
        if (result.index() == 1) {
            std::cerr << std::get<std::string>(result) << '\n';
            return false;
        }

        m_cachedEncoderOutputs = std::move(std::get<0>(result));
        m_cachedEncoderHostCopy = CachedEncoderOutputs();
        m_hasCachedEncoderHostCopy = false;
        return true;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] preprocessImage => " << error.what() << '\n';
        m_cachedEncoderOutputs.clear();
        m_cachedEncoderHostCopy = CachedEncoderOutputs();
        m_hasCachedEncoderHostCopy = false;
        return false;
    }
}

std::vector<float> SAM3::buildNoMemoryImageEmbedding(const Ort::Value& currentVisionFeat)
{
    std::vector<float> currentVisionValues;
    std::vector<int64_t> currentVisionShape;
    extractTensorData(currentVisionFeat, currentVisionValues, currentVisionShape);

    m_noMemoryImageEmbedScratch.resize(currentVisionValues.size());
    if (currentVisionValues.size() == m_constants.noMemEmbed.size()) {
        for (size_t index = 0; index < currentVisionValues.size(); ++index) {
            m_noMemoryImageEmbedScratch[index] = currentVisionValues[index] + m_constants.noMemEmbed[index];
        }
        return m_noMemoryImageEmbedScratch;
    }

    if (currentVisionShape.size() != m_constants.noMemEmbedShape.size()
        || m_constants.noMemEmbed.empty()) {
        throw std::runtime_error("current_vision_feat and no_mem_embed_bchw shapes are not broadcast-compatible.");
    }

    std::vector<size_t> noMemStrides(m_constants.noMemEmbedShape.size(), 1);
    for (size_t dim = m_constants.noMemEmbedShape.size(); dim-- > 1;) {
        noMemStrides[dim - 1] = noMemStrides[dim] * static_cast<size_t>(m_constants.noMemEmbedShape[dim]);
    }

    for (size_t dim = 0; dim < currentVisionShape.size(); ++dim) {
        const int64_t currentExtent = currentVisionShape[dim];
        const int64_t noMemExtent = m_constants.noMemEmbedShape[dim];
        if (currentExtent <= 0 || noMemExtent <= 0 || (noMemExtent != 1 && noMemExtent != currentExtent)) {
            throw std::runtime_error("current_vision_feat and no_mem_embed_bchw shapes are not broadcast-compatible.");
        }
    }

    std::vector<size_t> coords(currentVisionShape.size(), 0);
    for (size_t currentIndex = 0; currentIndex < currentVisionValues.size(); ++currentIndex) {
        size_t remaining = currentIndex;
        for (size_t dim = currentVisionShape.size(); dim-- > 0;) {
            const size_t extent = static_cast<size_t>(currentVisionShape[dim]);
            coords[dim] = remaining % extent;
            remaining /= extent;
        }

        size_t noMemIndex = 0;
        for (size_t dim = 0; dim < currentVisionShape.size(); ++dim) {
            const size_t sourceCoord = m_constants.noMemEmbedShape[dim] == 1 ? 0 : coords[dim];
            noMemIndex += sourceCoord * noMemStrides[dim];
        }

        m_noMemoryImageEmbedScratch[currentIndex] =
            currentVisionValues[currentIndex] + m_constants.noMemEmbed[noMemIndex];
    }
    return m_noMemoryImageEmbedScratch;
}

bool SAM3::hasCudaDriver()
{
    static int cached = -1;
    if (cached != -1) {
        return cached != 0;
    }

#if defined(_WIN32)
    if (!preloadCudaWindowsRuntime()) {
        cached = 0;
        return false;
    }

    static HMODULE cudartHandle = GetModuleHandleW(L"cudart64_12.dll");
    if (!cudartHandle) {
        cudartHandle = loadDllFromSearchDirectories(L"cudart64_12.dll", collectCudaSearchDirectories());
    }
    if (!cudartHandle) {
        cached = 0;
        return false;
    }

    using GetCount = int(__cdecl*)(int*);
    const auto function = reinterpret_cast<GetCount>(GetProcAddress(cudartHandle, "cudaGetDeviceCount"));
    if (!function) {
        cached = 0;
        return false;
    }

    int deviceCount = 0;
    cached = (function(&deviceCount) == 0 && deviceCount > 0) ? 1 : 0;
    return cached != 0;
#elif defined(__linux__)
    void* cudartHandle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_LOCAL);
    if (!cudartHandle) {
        cached = 0;
        return false;
    }

    using GetCount = int (*)(int*);
    const auto function = reinterpret_cast<GetCount>(dlsym(cudartHandle, "cudaGetDeviceCount"));
    if (!function) {
        cached = 0;
        return false;
    }

    int deviceCount = 0;
    cached = (function(&deviceCount) == 0 && deviceCount > 0) ? 1 : 0;
    return cached != 0;
#else
    cached = 0;
    return false;
#endif
}
