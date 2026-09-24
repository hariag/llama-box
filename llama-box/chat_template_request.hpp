#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

struct llama_box_chat_template_request_options {
    bool                                  enable_thinking = true;
    std::map<std::string, std::string>    template_kwargs;
};

// OpenAI's object form of tool_choice restricts generation to the named
// function.  An empty name means no restriction (the string forms are
// handled by the caller).
static inline bool llama_box_tool_matches_choice(
        const std::string & tool_name, const std::string & selected_name) {
    return selected_name.empty() || tool_name == selected_name;
}

// Upstream's chat parser consumes the whitespace separating a reasoning block
// from the final answer.  The server's token-level streaming path needs the
// same stateful behavior so that it does not expose the template delimiter as
// assistant content.
struct llama_box_reasoning_transition_filter {
    bool    pending          = false;
    int32_t end_decode_round = -1;

    void begin(int32_t decode_round = -1) {
        pending          = true;
        end_decode_round = decode_round;
    }

    // MTP may expose content tokens in the same processed batch as </think>.
    // They must be held until the accumulated reasoning delta is emitted.
    bool in_same_batch(int32_t decode_round) const {
        return pending && end_decode_round >= 0 && decode_round == end_decode_round;
    }

    bool transition_ready(int32_t decode_round) const {
        return pending && (end_decode_round < 0 || decode_round != end_decode_round);
    }

    void complete() {
        pending          = false;
        end_decode_round = -1;
    }

    bool is_delimiter_whitespace(const std::string & piece) const {
        return pending && !piece.empty() && std::all_of(piece.begin(), piece.end(), [](char ch) {
            return std::isspace(static_cast<unsigned char>(ch)) != 0;
        });
    }

    std::string consume(const std::string & piece) {
        if (!pending || piece.empty()) {
            return piece;
        }

        const bool whitespace_only = std::all_of(piece.begin(), piece.end(), [](char ch) {
            return std::isspace(static_cast<unsigned char>(ch)) != 0;
        });
        if (whitespace_only) {
            return {};
        }

        pending = false;
        return piece;
    }
};

static inline bool llama_box_jinja_tool_call_has_envelope(const std::string & generated_text) {
    return generated_text.find("<tool_call>") != std::string::npos ||
           generated_text.find("<function=") != std::string::npos ||
           // gemma4 emits `<|tool_call>call:name{...}<tool_call|>`
           generated_text.find("<|tool_call>") != std::string::npos;
}

// The PEG parser can expose a partial tool call while the model is still
// emitting its XML envelope.  Treating that partial AST as final truncates
// arguments (for example, returning "{" instead of the complete JSON/XML
// argument payload).  Only stop early after a recognized envelope is closed.
static inline bool llama_box_jinja_tool_call_is_complete(const std::string & generated_text) {
    const bool has_tool_call_end = generated_text.find("</tool_call>") != std::string::npos;
    const bool has_function = generated_text.find("<function=") != std::string::npos;
    const bool has_function_end = generated_text.find("</function>") != std::string::npos;
    // gemma4's tool-call envelope closes with `<tool_call|>`
    const bool has_gemma4_end = generated_text.find("<tool_call|>") != std::string::npos;

    if (has_gemma4_end) {
        return true;
    }

    // Qwen3.8/Qwen3.5's official parser accepts an omitted opening
    // <tool_call>, but the closing envelope remains mandatory.  Requiring
    // both the function and outer closing tags prevents a partial XML AST
    // from being reported as a completed tool call.
    return has_tool_call_end && has_function && has_function_end;
}

// A chat template may put the opening reasoning tag in the generation prompt.
// In that case the first sampled token is already inside the reasoning block.
static inline bool llama_box_reasoning_is_prefilled(
        const std::string & generation_prompt,
        const std::string & reasoning_start_tag,
        const std::string & reasoning_end_tag) {
    if (generation_prompt.empty() || reasoning_start_tag.empty()) {
        return false;
    }
    const size_t start = generation_prompt.rfind(reasoning_start_tag);
    if (start == std::string::npos) {
        return false;
    }
    const size_t end = reasoning_end_tag.empty() ? std::string::npos : generation_prompt.rfind(reasoning_end_tag);
    return end == std::string::npos || start > end;
}

static inline std::string llama_box_parse_message_reasoning_content(const nlohmann::ordered_json & message) {
    if (!message.contains("reasoning_content") || message.at("reasoning_content").is_null()) {
        return {};
    }
    if (!message.at("reasoning_content").is_string()) {
        throw std::invalid_argument(
            "Illegal param: \"reasoning_content\" must be a string or null");
    }
    return message.at("reasoning_content").get<std::string>();
}

static inline llama_box_chat_template_request_options llama_box_parse_chat_template_request_options(
        const nlohmann::ordered_json & request,
        const bool default_enable_thinking,
        const std::map<std::string, std::string> & default_template_kwargs) {
    llama_box_chat_template_request_options options;
    options.enable_thinking = default_enable_thinking;
    options.template_kwargs = default_template_kwargs;

    if (request.contains("chat_template_kwargs")) {
        const nlohmann::ordered_json & request_kwargs = request.at("chat_template_kwargs");
        if (!request_kwargs.is_object()) {
            throw std::invalid_argument("Illegal param: \"chat_template_kwargs\" must be an object");
        }
        for (const auto & item : request_kwargs.items()) {
            options.template_kwargs[item.key()] = item.value().dump();
        }
    }

    const auto enable_thinking = options.template_kwargs.find("enable_thinking");
    if (enable_thinking != options.template_kwargs.end()) {
        if (enable_thinking->second == "true") {
            options.enable_thinking = true;
        } else if (enable_thinking->second == "false") {
            options.enable_thinking = false;
        } else if (!enable_thinking->second.empty() && enable_thinking->second[0] == '"') {
            throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
        }
    }

    if (request.contains("reasoning_effort") && !request.at("reasoning_effort").is_null()) {
        if (!request.at("reasoning_effort").is_string()) {
            throw std::invalid_argument("Illegal param: \"reasoning_effort\" must be a string");
        }
        const std::string reasoning_effort = request.at("reasoning_effort").get<std::string>();
        if (reasoning_effort == "none") {
            options.enable_thinking = false;
            options.template_kwargs["enable_thinking"] = "false";
            // OpenAI uses `none` as a transport-level switch.  It is not a
            // Qwen3.8 template value (that template accepts xhigh/medium/low
            // only), so do not pass it into Jinja.
            options.template_kwargs.erase("reasoning_effort");
        } else {
            options.template_kwargs["reasoning_effort"] = request.at("reasoning_effort").dump();
        }
    }

    const auto preserve_thinking = options.template_kwargs.find("preserve_thinking");
    const auto preserve_reasoning = options.template_kwargs.find("preserve_reasoning");
    if (preserve_thinking != options.template_kwargs.end() &&
        preserve_reasoning == options.template_kwargs.end()) {
        options.template_kwargs["preserve_reasoning"] = preserve_thinking->second;
    } else if (preserve_reasoning != options.template_kwargs.end() &&
               preserve_thinking == options.template_kwargs.end()) {
        options.template_kwargs["preserve_thinking"] = preserve_reasoning->second;
    }

    return options;
}
