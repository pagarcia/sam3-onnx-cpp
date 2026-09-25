#pragma once
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace smseg_sam3 {
// Only immutable encoder features/position constants are eligible. The owner
// invalidates entries whenever features are replaced, even if an allocator
// happens to reuse their host addresses. Tracker history is deliberately excluded.
class SAM3CudaInputCache {
#ifdef _WIN32
    using Copy = int(__stdcall*)(void*, const void*, size_t, int);
    using Select = int(__stdcall*)(int);
#else
    using Copy = int(*)(void*, const void*, size_t, int);
    using Select = int(*)(int);
#endif
    struct Entry {
        const void* source = nullptr;
        std::vector<int64_t> shape;
        Ort::Value tensor{nullptr};
    };
    Ort::MemoryInfo memory_;
    Ort::Allocator allocator_;
    std::array<Entry, 4> entries_;
    Copy copy_ = nullptr;
    Select select_ = nullptr;
    int device_ = 0;
public:
    uint64_t hits = 0, uploaded_bytes = 0;
    SAM3CudaInputCache(const Ort::Session& session, int device)
        : memory_("Cuda", OrtDeviceAllocator, device, OrtMemTypeDefault),
          allocator_(session, memory_), device_(device) {
#ifdef _WIN32
        HMODULE runtime = GetModuleHandleW(L"cudart64_13.dll");
        if (!runtime) runtime = GetModuleHandleW(L"cudart64_12.dll");
        if (runtime) {
            copy_ = reinterpret_cast<Copy>(GetProcAddress(runtime, "cudaMemcpy"));
            select_ = reinterpret_cast<Select>(GetProcAddress(runtime, "cudaSetDevice"));
        }
#elif defined(__linux__)
        // The CUDA provider owns the runtime for longer than this cache.
        copy_ = reinterpret_cast<Copy>(dlsym(RTLD_DEFAULT, "cudaMemcpy"));
        select_ = reinterpret_cast<Select>(dlsym(RTLD_DEFAULT, "cudaSetDevice"));
#endif
        if (!copy_ || !select_) throw std::runtime_error("CUDA input-cache copy functions unavailable");
    }
    void clear() { for (auto& entry : entries_) entry = Entry{}; }
    const Ort::Value& get(const Ort::Value& host) {
        const auto type = host.GetTensorTypeAndShapeInfo();
        if (type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            host.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU)
            throw std::runtime_error("CUDA input cache requires host FP32 features");
        const auto source = host.GetTensorData<float>();
        const auto shape = type.GetShape();
        for (auto& entry : entries_) {
            if (entry.source == source && entry.shape == shape) { ++hits; return entry.tensor; }
        }
        Entry* slot = nullptr;
        for (auto& entry : entries_) if (!entry.source) { slot = &entry; break; }
        if (!slot) throw std::runtime_error("CUDA input cache exceeds its four-feature bound");
        const auto bytes = type.GetElementCount() * sizeof(float);
        if (bytes > 32ULL * 1024 * 1024) throw std::runtime_error("Unexpectedly large CUDA feature tensor");
        if (select_(device_) != 0) throw std::runtime_error("Cannot select CUDA feature device");
        auto tensor = Ort::Value::CreateTensor<float>(allocator_, shape.data(), shape.size());
        // Synchronous copy is intentional: the feature may be consumed by a
        // different ORT session/stream immediately after this method returns.
        if (copy_(tensor.GetTensorMutableData<float>(), source, bytes, 1) != 0)
            throw std::runtime_error("Cannot upload CUDA image features");
        slot->tensor = std::move(tensor); slot->shape = shape; slot->source = source;
        uploaded_bytes += bytes;
        return slot->tensor;
    }
};
} // namespace smseg_sam3
