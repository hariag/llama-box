#!/usr/bin/env python3
"""Exercise parallel tool calls against a separately started model server.

The tool definitions are synthetic; this script never executes a tool.
"""

import argparse
import json
import os
import urllib.request


GATEWAY = {
    "type": "function",
    "function": {
        "name": "mcp__tool-gateway__call_tool",
        "description": "调用指定的后端工具",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "arguments": {"type": "object", "additionalProperties": True},
            },
            "required": ["id", "arguments"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "mcp__tool-gateway__search_tools",
        "description": "查找工具",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
SIMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    for name, description in (("get_weather", "Query weather"), ("get_time", "Query time"))
]


def complete(server, endpoint, model, *, stream, parallel, seed, simple_tools):
    prompt = (
        "Call both functions: get_weather for Beijing and get_time for Shanghai."
        if simple_tools else
        "请使用 call_tool 调用两个工具：id 为 memory.memory_list_todos 的工具查询个人待办，"
        "以及 id 为 minutes.list_my_followups 的工具查询会议跟进。先调用这两个工具，然后等待结果。"
    )
    tools = SIMPLE_TOOLS if simple_tools else [GATEWAY, SEARCH]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "parallel_tool_calls": parallel,
        "stream": stream,
        "temperature": 0.5,
        "seed": seed,
        "max_completion_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLAMA_BOX_TEST_AUTH"):
        headers["Authorization"] = os.environ["LLAMA_BOX_TEST_AUTH"]
    request = urllib.request.Request(server.rstrip("/") + endpoint,
                                     json.dumps(body, ensure_ascii=False).encode(), headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        raw = response.read().decode("utf-8", errors="strict")
    if stream:
        lines = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
        assert lines and lines[-1] == "[DONE]", raw
        records = [json.loads(line) for line in lines[:-1]]
    else:
        records = [json.loads(raw)]

    content, calls, reasons = [], [], []
    for record in records:
        for choice in record.get("choices", []):
            message = choice.get("delta", {}) if stream else choice.get("message", {})
            content.append(message.get("content") or "")
            calls.extend(message.get("tool_calls") or [])
            if choice.get("finish_reason") is not None:
                reasons.append(choice["finish_reason"])

    text = "".join(content)
    assert not any(marker in text for marker in (
        "<tool_call>", "</tool_call>", "<function=", "</function>", "<|tool_call>", "<tool_call|>",
    )), text
    assert calls, (text, reasons)
    assert reasons == ["tool_calls"], (text, calls, reasons)
    if stream:
        assert [call.get("index") for call in calls] == list(range(len(calls))), calls
    for call in calls:
        assert call.get("type") == "function" and call.get("id"), call
        function = call["function"]
        assert function["name"] in {tool["function"]["name"] for tool in tools}, function
        assert isinstance(json.loads(function["arguments"]), dict), function
    return len(calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="llama-box base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--simple-tools", action="store_true", help="use two simple tools for small models")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    args = parser.parse_args()

    parallel_results = []
    for stream in (True, False):
        for seed in range(args.iterations):
            count = complete(args.server, args.endpoint, args.model, stream=stream, parallel=True, seed=seed,
                             simple_tools=args.simple_tools)
            parallel_results.append(count)
            print("PASS", "stream" if stream else "non-stream", "seed", seed, "calls", count, flush=True)
    assert max(parallel_results) >= 2, "the model never produced parallel calls"
    for stream in (True, False):
        count = complete(args.server, args.endpoint, args.model, stream=stream, parallel=False, seed=0,
                         simple_tools=args.simple_tools)
        assert count == 1, count
        print("PASS", "stream" if stream else "non-stream", "single call", flush=True)


if __name__ == "__main__":
    main()
