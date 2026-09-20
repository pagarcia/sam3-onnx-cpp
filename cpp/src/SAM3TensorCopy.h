#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace smseg_sam3 {

// Core ML may return padded strides. Bounds are checked before reading the buffer.
inline void copyStridedFloatTensor(const float* source, size_t sourceCount,
                                  const std::vector<size_t>& shape,
                                  const std::vector<size_t>& strides,
                                  float* destination, size_t destinationCount)
{
    if (!source || !destination || shape.empty() || shape.size() != strides.size())
        throw std::runtime_error("Invalid tensor buffer or rank");
    size_t count = 1, last = 0;
    bool contiguous = true;
    for (size_t axis = shape.size(); axis-- > 0;) {
        const size_t dim = shape[axis], stride = strides[axis];
        if (!dim || !stride || dim > std::numeric_limits<size_t>::max() / count
            || dim - 1 > (std::numeric_limits<size_t>::max() - last) / stride)
            throw std::runtime_error("Invalid tensor shape or strides");
        if (dim > 1 && stride != count) contiguous = false;
        last += (dim - 1) * stride;
        count *= dim;
    }
    if (count != destinationCount || last >= sourceCount)
        throw std::runtime_error("Tensor buffer size mismatch");
    if (contiguous) {
        std::copy_n(source, count, destination);
    } else {
        for (size_t flat = 0; flat < count; ++flat) {
            size_t remaining = flat, offset = 0;
            for (size_t axis = shape.size(); axis-- > 0;) {
                offset += remaining % shape[axis] * strides[axis];
                remaining /= shape[axis];
            }
            destination[flat] = source[offset];
        }
    }
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(destination[i]))
            throw std::runtime_error("Nonfinite encoder output");
}

} // namespace smseg_sam3
