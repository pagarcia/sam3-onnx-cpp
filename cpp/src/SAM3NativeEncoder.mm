#include "SAM3NativeEncoder.h"
#include "SAM3TensorCopy.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <array>
#include <filesystem>

namespace smseg_sam3 {
namespace {
const std::array<std::vector<int64_t>, 3> outputShapes{{
    {1, 32, 288, 288}, {1, 64, 144, 144}, {1, 256, 72, 72}}};

std::runtime_error error(const char* operation, NSError* detail)
{
    return std::runtime_error(std::string(operation) + ": "
        + (detail ? detail.localizedDescription.UTF8String : "Core ML returned no result"));
}

void requireShape(MLMultiArrayConstraint* constraint, const std::vector<int64_t>& expected)
{
    if (!constraint || constraint.dataType != MLMultiArrayDataTypeFloat32
        || constraint.shape.count != expected.size())
        throw std::runtime_error("Unexpected Core ML tensor type or rank");
    for (size_t i = 0; i < expected.size(); ++i)
        if (constraint.shape[i].longLongValue != expected[i])
            throw std::runtime_error("Unexpected Core ML tensor shape");
}

class Encoder final : public SAM3NativeEncoder {
    MLModel* model_ = nil;
public:
    Encoder(const std::string& path, bool cpuOnly)
    {
        @autoreleasepool {
            const std::filesystem::path modelPath = std::filesystem::path(path);
            if (modelPath.extension() != ".mlmodelc" || !std::filesystem::is_directory(modelPath))
                throw std::runtime_error("Expected a compiled .mlmodelc encoder directory");
            NSString* name = [[NSString alloc] initWithBytes:path.data() length:path.size()
                                                   encoding:NSUTF8StringEncoding];
            if (!name) throw std::runtime_error("Invalid UTF-8 model path");
            MLModelConfiguration* config = [MLModelConfiguration new];
            config.computeUnits = cpuOnly ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndGPU;
            config.allowLowPrecisionAccumulationOnGPU = NO;
            NSError* detail = nil;
            model_ = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:name]
                                      configuration:config error:&detail];
            if (!model_) throw error("Loading native encoder", detail);
            auto inputs = model_.modelDescription.inputDescriptionsByName;
            auto outputs = model_.modelDescription.outputDescriptionsByName;
            if (inputs.count != 1 || outputs.count != 3)
                throw std::runtime_error("Unexpected Core ML encoder interface");
            requireShape(inputs[@"image"].multiArrayConstraint, {1, 3, 1008, 1008});
            for (size_t i = 0; i < 3; ++i) {
                NSString* output = [NSString stringWithFormat:@"embedding%zu", i];
                requireShape(outputs[output].multiArrayConstraint, outputShapes[i]);
            }
        }
    }

    std::vector<Ort::Value> run(const std::vector<float>& nchw) override
    {
        if (nchw.size() != 3 * 1008 * 1008)
            throw std::runtime_error("Expected one FP32 1008-square RGB input");
        for (float value : nchw)
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite encoder input");
        @autoreleasepool {
            NSError* detail = nil;
            // Prediction is synchronous; the caller owns this buffer throughout the call.
            MLMultiArray* input = [[MLMultiArray alloc]
                initWithDataPointer:const_cast<float*>(nchw.data())
                shape:@[@1, @3, @1008, @1008] dataType:MLMultiArrayDataTypeFloat32
                strides:@[@3048192, @1016064, @1008, @1]
                deallocator:^(void*) {} error:&detail];
            if (!input) throw error("Creating encoder input", detail);
            MLDictionaryFeatureProvider* feed = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"image": [MLFeatureValue featureValueWithMultiArray:input]}
                error:&detail];
            if (!feed) throw error("Creating encoder features", detail);
            id<MLFeatureProvider> prediction = [model_ predictionFromFeatures:feed error:&detail];
            if (!prediction) throw error("Running native encoder", detail);
            std::vector<Ort::Value> result;
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < 3; ++i) {
                NSString* name = [NSString stringWithFormat:@"embedding%zu", i];
                MLMultiArray* array = [prediction featureValueForName:name].multiArrayValue;
                const auto& dimensions = outputShapes[i];
                if (!array || array.dataType != MLMultiArrayDataTypeFloat32
                    || array.shape.count != dimensions.size())
                    throw std::runtime_error("Unexpected native encoder output");
                std::vector<size_t> shape, strides;
                size_t count = 1;
                for (size_t d = 0; d < dimensions.size(); ++d) {
                    if (array.shape[d].longLongValue != dimensions[d]
                        || array.strides[d].longLongValue <= 0)
                        throw std::runtime_error("Invalid native encoder output dimensions");
                    shape.push_back(static_cast<size_t>(dimensions[d]));
                    strides.push_back(array.strides[d].unsignedLongLongValue);
                    count *= shape.back();
                }
                auto tensor = Ort::Value::CreateTensor<float>(allocator, dimensions.data(), dimensions.size());
                float* destination = tensor.GetTensorMutableData<float>();
                [array getBytesWithHandler:^(const void* bytes, NSInteger size) {
                    if (size < 0) throw std::runtime_error("Invalid native encoder output buffer");
                    copyStridedFloatTensor(static_cast<const float*>(bytes),
                        static_cast<size_t>(size) / sizeof(float), shape, strides, destination, count);
                }];
                result.push_back(std::move(tensor));
            }
            return result;
        }
    }
};
} // namespace

std::unique_ptr<SAM3NativeEncoder> SAM3NativeEncoder::create(const std::string& path, bool cpuOnly)
{
    if (@available(macOS 15.0, *)) return std::make_unique<Encoder>(path, cpuOnly);
    throw std::runtime_error("Native encoder requires macOS 15 or later");
}
} // namespace smseg_sam3
