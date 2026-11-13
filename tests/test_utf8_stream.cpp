#include "../llama-box/z_utf8_stream.hpp"
#include "../llama.cpp/vendor/nlohmann/json.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    // Enumerate all token partitions, including boundaries inside 3- and
    // 4-byte characters and tokens containing parts of adjacent characters.
    const std::string text = u8"A✅中🙂B";
    for (uint64_t mask = 0; mask < (uint64_t{1} << (text.size() - 1)); ++mask) {
        std::vector<size_t> token_ends;
        for (size_t pos = 1; pos < text.size(); ++pos) {
            if (mask & (uint64_t{1} << (pos - 1))) token_ends.push_back(pos);
        }
        token_ends.push_back(text.size());
        const auto lengths = utf8_stream_chunk_lengths(text, token_ends);
        assert(!lengths.empty());
        std::string joined;
        size_t pos = 0;
        for (const auto len : lengths) {
            const auto part = text.substr(pos, len);
            const auto encoded = nlohmann::json{{"content", part}}.dump();
            joined += nlohmann::json::parse(encoded).at("content").get<std::string>();
            pos += len;
        }
        assert(pos == text.size());
        assert(joined == text);
    }
    assert(utf8_stream_chunk_lengths("abc", {1, 4}).empty());
    assert(utf8_stream_chunk_lengths("abc", {2, 1, 3}).empty());
    assert(utf8_stream_chunk_lengths("abc", {1, 2}).empty());
}
