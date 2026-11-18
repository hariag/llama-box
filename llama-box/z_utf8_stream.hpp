#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

static inline bool string_is_utf8(std::string_view text) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(text.data());
    for (size_t pos = 0; pos < text.size();) {
        const auto lead = bytes[pos];
        if (lead <= 0x7f) {
            ++pos;
            continue;
        }

        size_t width;
        if (lead >= 0xc2 && lead <= 0xdf) {
            width = 2;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            width = 3;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            width = 4;
        } else {
            return false;
        }
        if (text.size() - pos < width) {
            return false;
        }
        const auto second = bytes[pos + 1];
        // Reject overlong sequences, surrogate code points and values above U+10FFFF.
        if ((lead == 0xe0 && second < 0xa0) || (lead == 0xed && second >= 0xa0) ||
            (lead == 0xf0 && second < 0x90) || (lead == 0xf4 && second >= 0x90)) {
            return false;
        }
        for (size_t offset = 1; offset < width; ++offset) {
            if ((bytes[pos + offset] & 0xc0) != 0x80) {
                return false;
            }
        }
        pos += width;
    }
    return true;
}

static inline std::string utf8_format_token_piece(std::string piece) {
    if (string_is_utf8(piece)) {
        return piece;
    }

    // A token can be a prefix or suffix of a code point even when it has
    // several bytes. Label it without letting JSON replace the token bytes.
    constexpr char hex_digits[] = "0123456789abcdef";
    std::string escaped = piece.size() == 1 ? "byte: " : "bytes: ";
    for (unsigned char byte : piece) {
        escaped += "\\x";
        escaped += hex_digits[byte >> 4];
        escaped += hex_digits[byte & 0x0f];
    }
    return escaped;
}

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
