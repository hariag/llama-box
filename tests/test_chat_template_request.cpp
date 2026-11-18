#include <cassert>
#include <stdexcept>

#include "llama-box/chat_template_request.hpp"

static void test_request_kwargs_override_defaults() {
    nlohmann::ordered_json request = {
        { "chat_template_kwargs", {
            { "enable_thinking", true },
            { "preserve_thinking", true },
        } },
        { "reasoning_effort", "xhigh" },
    };

    const std::map<std::string, std::string> defaults = {
        { "enable_thinking", "false" },
        { "server_default", "42" },
    };

    const auto options = llama_box_parse_chat_template_request_options(request, false, defaults);

    assert(options.enable_thinking);
    assert(options.template_kwargs.at("enable_thinking") == "true");
    assert(options.template_kwargs.at("preserve_thinking") == "true");
    assert(options.template_kwargs.at("reasoning_effort") == "\"xhigh\"");
    assert(options.template_kwargs.at("server_default") == "42");
    assert(options.template_kwargs.at("preserve_reasoning") == "true");
}

static void test_reasoning_effort_none_disables_thinking() {
    nlohmann::ordered_json request = {
        { "chat_template_kwargs", {{ "enable_thinking", true }} },
        { "reasoning_effort", "none" },
    };
    const auto options = llama_box_parse_chat_template_request_options(request, true, {});

    assert(!options.enable_thinking);
    assert(options.template_kwargs.at("enable_thinking") == "false");
    assert(options.template_kwargs.find("reasoning_effort") == options.template_kwargs.end());
}

static void test_explicit_disable_wins_without_reasoning_effort() {
    nlohmann::ordered_json request = {{ "chat_template_kwargs", {{ "enable_thinking", false }} }};
    const auto options = llama_box_parse_chat_template_request_options(request, true, {});

    assert(!options.enable_thinking);
    assert(options.template_kwargs.at("enable_thinking") == "false");
}

static void test_reasoning_effort_levels_are_forwarded() {
    for (const char * effort : { "xhigh", "medium", "low" }) {
        nlohmann::ordered_json request = {{ "reasoning_effort", effort }};
        const auto options = llama_box_parse_chat_template_request_options(request, true, {});
        assert(options.enable_thinking);
        assert(options.template_kwargs.at("reasoning_effort") == nlohmann::ordered_json(effort).dump());
    }
}

static void test_invalid_chat_template_kwargs_is_rejected() {
    nlohmann::ordered_json request = {{ "chat_template_kwargs", true }};
    bool rejected = false;
    try {
        (void) llama_box_parse_chat_template_request_options(request, true, {});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

static void test_invalid_reasoning_effort_is_rejected() {
    nlohmann::ordered_json request = {{ "reasoning_effort", 3 }};
    bool rejected = false;
    try {
        (void) llama_box_parse_chat_template_request_options(request, true, {});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

static void test_preserve_reasoning_alias_is_bidirectional() {
    const auto from_thinking = llama_box_parse_chat_template_request_options(
        nlohmann::ordered_json{{ "chat_template_kwargs", {{ "preserve_thinking", true }} }}, true, {});
    assert(from_thinking.template_kwargs.at("preserve_thinking") == "true");
    assert(from_thinking.template_kwargs.at("preserve_reasoning") == "true");

    const auto from_reasoning = llama_box_parse_chat_template_request_options(
        nlohmann::ordered_json{{ "chat_template_kwargs", {{ "preserve_reasoning", false }} }}, true, {});
    assert(from_reasoning.template_kwargs.at("preserve_reasoning") == "false");
    assert(from_reasoning.template_kwargs.at("preserve_thinking") == "false");
}

static void test_message_reasoning_content_is_preserved() {
    const nlohmann::ordered_json message = {
        { "role", "assistant" },
        { "content", "answer" },
        { "reasoning_content", "internal marker" },
    };
    assert(llama_box_parse_message_reasoning_content(message) == "internal marker");
    assert(llama_box_parse_message_reasoning_content(
               nlohmann::ordered_json{{ "role", "assistant" }, { "content", "answer" }})
           .empty());
    assert(llama_box_parse_message_reasoning_content(
               nlohmann::ordered_json{{ "role", "assistant" }, { "content", "answer" }, { "reasoning_content", nullptr }})
           .empty());
    assert(llama_box_parse_message_reasoning_content(nlohmann::ordered_json{
               { "role", "assistant" },
               { "tool_calls", nlohmann::ordered_json::array({
                                   {{ "type", "function" },
                                    { "function", {{ "name", "lookup" }, { "arguments", "{}" }} }},
                               }) },
               { "reasoning_content", "tool reasoning" },
           }) == "tool reasoning");
}

static void test_invalid_message_reasoning_content_is_rejected() {
    bool rejected = false;
    try {
        (void) llama_box_parse_message_reasoning_content(
            nlohmann::ordered_json{{ "role", "assistant" }, { "content", "answer" }, { "reasoning_content", 1 }});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

static void test_jinja_tool_call_requires_a_closed_envelope_before_stop() {
    assert(!llama_box_jinja_tool_call_is_complete("<tool_call>\n<function=get_weather>\n"));
    assert(!llama_box_jinja_tool_call_is_complete("<function=get_weather>\n<parameter=location>\n"));
    assert(llama_box_jinja_tool_call_is_complete(
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nBeijing\n"
        "</parameter>\n</function>\n</tool_call>"));
    assert(!llama_box_jinja_tool_call_is_complete(
        "<function=get_weather>\n<parameter=location>\nBeijing\n</parameter>\n</function>"));
    assert(llama_box_jinja_tool_call_has_envelope("<function=get_weather>"));
    assert(!llama_box_jinja_tool_call_has_envelope("ordinary response"));
}

static void test_jinja_tool_call_spans_preserve_following_calls() {
    const std::string first =
        "<tool_call>\n<function=first>\n<parameter=id>\none\n</parameter>\n</function>\n</tool_call>";
    const std::string second =
        "<tool_call>\n<function=second>\n<parameter=id>\ntwo\n</parameter>\n</function>\n</tool_call>";
    const std::string text = "before " + first + "\n" + second.substr(0, 24);
    const auto first_span = llama_box_find_jinja_tool_call(text);
    assert(first_span.start == 7);
    assert(first_span.end == 7 + first.size());
    assert(!first_span.missing_open);

    const std::string rest = text.substr(first_span.end);
    const auto next_span = llama_box_find_jinja_tool_call(rest);
    assert(next_span.start == 1);
    assert(next_span.end == std::string::npos);

    const auto missing_open = llama_box_find_jinja_tool_call(second.substr(std::string("<tool_call>\n").size()));
    assert(missing_open.start == 0);
    assert(missing_open.end != std::string::npos);
    assert(missing_open.missing_open);

    const std::string gemma = "prefix <|tool_call>call:lookup{}<tool_call|> tail";
    const auto gemma_span = llama_box_find_jinja_tool_call(gemma);
    assert(gemma_span.start == 7);
    assert(gemma.substr(gemma_span.end) == " tail");
}

static void test_gemma4_tool_call_ends_unclosed_thought() {
    const std::string thought = "Checking the weather";
    const std::string tool = "<|tool_call>";
    for (size_t n = 1; n < tool.size(); ++n) {
        const auto partial = llama_box_find_gemma4_thought_boundary(thought + tool.substr(0, n), true);
        assert(partial.reasoning_end == thought.size());
        assert(partial.content_start == std::string::npos);
    }
    const auto unclosed = llama_box_find_gemma4_thought_boundary(thought + tool, true);
    assert(unclosed.reasoning_end == thought.size());
    assert(unclosed.content_start == thought.size());
    const auto closed = llama_box_find_gemma4_thought_boundary(thought + "<channel|>" + tool, true);
    assert(closed.reasoning_end == thought.size());
    assert(closed.content_start == thought.size() + std::string_view("<channel|>").size());
    const auto disabled = llama_box_find_gemma4_thought_boundary(thought + tool, false);
    assert(disabled.reasoning_end == thought.size() + tool.size());
    assert(disabled.content_start == std::string::npos);
}

static void test_jinja_tool_call_partial_markers_are_buffered() {
    for (const std::string marker : { "<tool_call>", "<function=", "<|tool_call>" }) {
        for (size_t split = 1; split < marker.size(); ++split) {
            const std::string text = "answer " + marker.substr(0, split);
            assert(llama_box_jinja_tool_call_partial_start(text) == 7);
        }
    }
    const std::string text = "answer <ordinary text";
    assert(llama_box_jinja_tool_call_partial_start(text) == text.size());
}

static void test_prefilled_reasoning_prompt_state() {
    assert(llama_box_reasoning_is_prefilled(
        "<|im_start|>assistant\n<think>\n", "<think>", "</think>"));
    assert(!llama_box_reasoning_is_prefilled(
        "<|im_start|>assistant\n<think>\n</think>\n", "<think>", "</think>"));
    assert(!llama_box_reasoning_is_prefilled(
        "<|im_start|>assistant\n", "<think>", "</think>"));
}

static void test_reasoning_transition_suppresses_only_delimiter_whitespace() {
    llama_box_reasoning_transition_filter filter;
    filter.begin();

    assert(filter.consume("\n") == "");
    assert(filter.consume("\n") == "");
    assert(filter.consume("4") == "4");
    assert(filter.consume("\n") == "\n");

    filter.begin();
    assert(filter.consume(" 4") == " 4");
}

static void test_reasoning_transition_defers_same_batch_content() {
    llama_box_reasoning_transition_filter filter;
    filter.begin(10);

    assert(filter.in_same_batch(10));
    assert(filter.is_delimiter_whitespace("\n"));
    assert(!filter.is_delimiter_whitespace("4"));
    assert(filter.consume("\n").empty());
    assert(!filter.transition_ready(10));
    assert(filter.transition_ready(11));

    filter.complete();
    assert(!filter.transition_ready(11));
}

static void test_specific_tool_choice_keeps_the_selected_tool() {
    assert(llama_box_tool_matches_choice("get_weather", "get_weather"));
    assert(!llama_box_tool_matches_choice("get_time", "get_weather"));
    assert(llama_box_tool_matches_choice("get_time", ""));
}

int main() {
    test_request_kwargs_override_defaults();
    test_reasoning_effort_none_disables_thinking();
    test_explicit_disable_wins_without_reasoning_effort();
    test_reasoning_effort_levels_are_forwarded();
    test_invalid_chat_template_kwargs_is_rejected();
    test_invalid_reasoning_effort_is_rejected();
    test_preserve_reasoning_alias_is_bidirectional();
    test_message_reasoning_content_is_preserved();
    test_invalid_message_reasoning_content_is_rejected();
    test_jinja_tool_call_requires_a_closed_envelope_before_stop();
    test_jinja_tool_call_spans_preserve_following_calls();
    test_gemma4_tool_call_ends_unclosed_thought();
    test_jinja_tool_call_partial_markers_are_buffered();
    test_prefilled_reasoning_prompt_state();
    test_reasoning_transition_suppresses_only_delimiter_whitespace();
    test_reasoning_transition_defers_same_batch_content();
    test_specific_tool_choice_keeps_the_selected_tool();
    return 0;
}
