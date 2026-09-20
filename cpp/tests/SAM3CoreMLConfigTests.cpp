#include "SAM3CoreML.h"
#include <iostream>

using smseg_sam3::SAM3CoreMLConfig;

void require(bool condition)
{
    if (!condition) throw std::runtime_error("Core ML configuration assertion failed");
}

template<class Operation> void rejects(Operation operation)
{
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}

int main()
{
    try {
        SAM3CoreMLConfig config;
        auto options = config.providerOptions();
        require(options.at("ModelFormat") == "MLProgram");
        require(options.at("AllowLowPrecisionAccumulationOnGPU") == "0");
        require(!options.count("ModelCacheDirectory"));
        config.computeUnits = "GPU";
        rejects([&] { config.providerOptions(); });
        config.computeUnits = "CPUAndGPU";
        config.specialization = "unknown";
        rejects([&] { config.providerOptions(); });
        config.specialization = "FastPrediction";
        config.cacheDirectory = std::filesystem::temp_directory_path().string();
        rejects([&] { config.providerOptions(); });
        config.cacheKey = "../unverified-cache";
        rejects([&] { config.providerOptions(); });
        config.cacheKey = std::string(64, 'a');
        require(config.providerOptions().at("ModelCacheDirectory") ==
                (std::filesystem::path(config.cacheDirectory) / config.cacheKey).string());
        config.cacheDirectory = "relative";
        rejects([&] { config.providerOptions(); });
        std::cout << "Core ML configuration checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
