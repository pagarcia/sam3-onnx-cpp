#include "SAM3TensorCopy.h"
#include <iostream>
#include <limits>

int main()
{
    using smseg_sam3::copyStridedFloatTensor;
    try {
        const float padded[]{1, 2, 0, 3, 4, 0};
        float output[4]{};
        copyStridedFloatTensor(padded, 6, {2, 2}, {3, 1}, output, 4);
        if (std::vector<float>(output, output + 4) != std::vector<float>{1, 2, 3, 4})
            throw std::runtime_error("Padded tensor copy changed values");
        copyStridedFloatTensor(padded, 6, {2, 2}, {1, 3}, output, 4);
        if (std::vector<float>(output, output + 4) != std::vector<float>{1, 3, 2, 4})
            throw std::runtime_error("Transposed tensor copy changed values");
        copyStridedFloatTensor(padded, 6, {1, 2}, {100, 1}, output, 2);
        if (output[0] != 1 || output[1] != 2)
            throw std::runtime_error("Singleton strides must not affect contiguous copies");
        auto rejected = [](auto operation) {
            try { operation(); } catch (const std::runtime_error&) { return; }
            throw std::runtime_error("Invalid tensor was accepted");
        };
        rejected([&] { copyStridedFloatTensor(padded, 4, {2, 2}, {3, 1}, output, 4); });
        rejected([&] { copyStridedFloatTensor(padded, 6, {2, 2}, {3, 1}, output, 3); });
        rejected([&] { copyStridedFloatTensor(padded, 6, {0, 2}, {3, 1}, output, 0); });
        rejected([&] { copyStridedFloatTensor(padded, 6, {2, 2}, {1}, output, 4); });
        rejected([&] { copyStridedFloatTensor(padded, 6, {2, 2}, {SIZE_MAX, 1}, output, 4); });
        float invalid[]{std::numeric_limits<float>::quiet_NaN()};
        rejected([&] { copyStridedFloatTensor(invalid, 1, {1}, {1}, output, 1); });
        std::cout << "Tensor copy checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
