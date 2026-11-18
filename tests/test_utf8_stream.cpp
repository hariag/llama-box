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

    // Token display strings must be valid JSON even when a BPE token spans
    // two bytes of a three-byte character. Check both split directions.
    const std::string character = u8"中";
    assert(utf8_format_token_piece(character) == character);
    assert(utf8_format_token_piece(character.substr(0, 1)) == "byte: \\xe4");
    assert(utf8_format_token_piece(character.substr(0, 2)) == "bytes: \\xe4\\xb8");
    assert(utf8_format_token_piece(character.substr(1)) == "bytes: \\xb8\\xad");
    assert(utf8_format_token_piece("a" + character.substr(0, 2)) == "bytes: \\x61\\xe4\\xb8");

    // These sequences have continuation bytes but cannot encode Unicode scalars.
    assert(!string_is_utf8("\xc0\x80"));
    assert(!string_is_utf8("\xed\xa0\x80"));
    assert(!string_is_utf8("\xf4\x90\x80\x80"));
    const auto piece = utf8_format_token_piece(character.substr(0, 2));
    assert(nlohmann::json::parse(nlohmann::json{{"token", piece}}.dump()).at("token") == piece);
}
