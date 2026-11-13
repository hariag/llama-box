#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

// The accumulated text has a complete UTF-8 ending before it is streamed.
// Token boundaries, however, can fall inside a code point. Merge those tokens
// so that no JSON string is serialized with a partial multi-byte character.
static inline std::vector<size_t> utf8_stream_chunk_lengths(
        std::string_view text, const std::vector<size_t> & token_ends) {
    std::vector<size_t> lengths;
    size_t last_token_end = 0;
    size_t last_chunk_end = 0;
    for (size_t end : token_ends) {
        if (end < last_token_end || end > text.size()) {
            return {};
        }
        last_token_end = end;
        if (end < text.size() && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80) {
            continue;
        }
        if (end > last_chunk_end) {
            lengths.push_back(end - last_chunk_end);
            last_chunk_end = end;
        }
    }
    if (last_chunk_end != text.size()) {
        return {};
    }
    return lengths;
}
