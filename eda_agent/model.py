from __future__ import annotations

import base64
import json
from typing import Any

from agentscope.formatter import AnthropicChatFormatter, GeminiChatFormatter, OpenAIChatFormatter
from agentscope.formatter._truncated_formatter_base import TruncatedFormatterBase
from agentscope.message import (
    Msg,
    TextBlock,
    ThinkingBlock,
    ImageBlock,
    AudioBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from agentscope.model import (
    AnthropicChatModel,
    ChatModelBase,
    GeminiChatModel,
    OpenAIChatModel,
)

from eda_agent.config import LLMConfig


class Llama4ChatFormatter(TruncatedFormatterBase):
    """Formatter for Llama-4 models that converts tool results to user messages.
    
    Llama-4 (via OpenAI-compatible APIs) does not support the "tool" role.
    This formatter converts tool_result blocks into user messages with formatted
    content instead of using the "tool" role.
    """

    support_tools_api: bool = True
    support_multiagent: bool = True
    support_vision: bool = True

    supported_blocks: list[type] = [
        TextBlock,
        ImageBlock,
        AudioBlock,
        ToolUseBlock,
        ToolResultBlock,
    ]

    async def _format(
        self,
        msgs: list[Msg],
    ) -> list[dict[str, Any]]:
        """Format message objects into Llama-4 compatible format.
        
        Tool results are converted to user messages instead of using "tool" role.
        """
        self.assert_list_of_msgs(msgs)

        messages: list[dict] = []
        for msg in msgs:
            content_blocks = []
            tool_calls = []
            tool_results = []

            for block in msg.get_content_blocks():
                typ = block.get("type")
                if typ == "text":
                    content_blocks.append({**block})

                elif typ == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(
                                    block.get("input", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        },
                    )

                elif typ == "tool_result":
                    # Collect tool results to convert to user message format
                    tool_name = block.get("name", "tool")
                    tool_id = block.get("id", "")
                    tool_output = self.convert_tool_result_to_string(
                        block.get("output"),  # type: ignore[arg-type]
                    )
                    tool_results.append({
                        "name": tool_name,
                        "id": tool_id,
                        "output": tool_output,
                    })

                elif typ == "image":
                    source_type = block["source"]["type"]
                    if source_type == "url":
                        url = block["source"]["url"]
                    elif source_type == "base64":
                        data = block["source"]["data"]
                        media_type = block["source"]["media_type"]
                        url = f"data:{media_type};base64,{data}"
                    else:
                        raise ValueError(
                            f"Unsupported image source type: {source_type}",
                        )

                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        },
                    )

                elif typ == "audio":
                    # Llama-4 may not support audio; skip or handle gracefully
                    pass

            # Build the message
            msg_dict = {
                "role": msg.role,
                "name": msg.name,
                "content": content_blocks or None,
            }

            if tool_calls:
                msg_dict["tool_calls"] = tool_calls

            # Add regular message if it has content or tool_calls
            if msg_dict["content"] or msg_dict.get("tool_calls"):
                messages.append(msg_dict)

            # Convert tool results to user messages (Llama-4 compatible format)
            for tr in tool_results:
                tool_result_text = (
                    f"[Tool Result: {tr['name']}]\n"
                    f"{tr['output']}"
                )
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": tool_result_text}],
                })

        return messages


class UsageTrackingModel(ChatModelBase):
    def __init__(self, base_model: ChatModelBase) -> None:
        self._base_model = base_model
        super().__init__(
            model_name=base_model.model_name,
            stream=base_model.stream,
        )
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self._base_model.model_name if hasattr(self, "_base_model") else self.__dict__.get("model_name", "")

    @model_name.setter
    def model_name(self, value: str) -> None:  # type: ignore[override]
        self.__dict__["model_name"] = value
        if hasattr(self, "_base_model"):
            self._base_model.model_name = value

    @property
    def stream(self) -> bool:  # type: ignore[override]
        return self._base_model.stream if hasattr(self, "_base_model") else bool(self.__dict__.get("stream", False))

    @stream.setter
    def stream(self, value: bool) -> None:
        self.__dict__["stream"] = value
        if hasattr(self, "_base_model"):
            self._base_model.stream = value

    @property
    def total_input_tokens(self) -> int:
        return self._input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._output_tokens

    def reset_usage(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0

    def _accumulate_usage(self, usage: Any) -> None:
        if usage is None:
            return
        self._input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self._output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    async def __call__(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        res = await self._base_model(*args, **kwargs)
        if self._base_model.stream:
            async def _gen():
                last_usage = None
                try:
                    async for chunk in res:
                        if getattr(chunk, "usage", None):
                            last_usage = chunk.usage
                        yield chunk
                finally:
                    # Streamed usage is cumulative (Anthropic even updates it in place after the
                    # last content chunk), so count it once, when the stream ends.
                    self._accumulate_usage(last_usage)
            return _gen()
        self._accumulate_usage(getattr(res, "usage", None))
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_model, name)


def get_model_usage(model: Any) -> tuple[int, int]:
    input_tokens = getattr(model, "total_input_tokens", None)
    output_tokens = getattr(model, "total_output_tokens", None)
    if input_tokens is None or output_tokens is None:
        return 0, 0
    return int(input_tokens), int(output_tokens)


def _is_llama_model(model_name: str) -> bool:
    """Check if the model name indicates a Llama model."""
    name_lower = model_name.lower()
    return any(k in name_lower for k in ["llama", "meta-llama"])


def _function_call_signatures(response: Any) -> list[bytes | None]:
    """Thought signatures of the function-call parts in a Gemini response, in order."""
    candidates = getattr(response, "candidates", None) or []
    content = getattr(candidates[0], "content", None) if candidates else None
    parts = getattr(content, "parts", None) or []
    return [getattr(p, "thought_signature", None) for p in parts if getattr(p, "function_call", None)]


def _attach_thought_signatures(blocks: list, signatures: list[bytes | None]) -> None:
    tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
    for block, signature in zip(tool_blocks, signatures):
        if signature:
            block["thought_signature"] = base64.b64encode(signature).decode("ascii")


class _GeminiChatModel(GeminiChatModel):
    """GeminiChatModel that keeps each tool call's thought signature.

    Gemini 3 models reject a request (HTTP 400) when a function call from the current turn is
    sent back without the `thought_signature` it came with. AgentScope 1.0.7 drops these, so we
    store each one (base64) on its ToolUseBlock and `_GeminiChatFormatter` sends it back.
    """

    def _parse_gemini_generation_response(self, start_datetime, response, structured_model=None):  # type: ignore[override]
        parsed = super()._parse_gemini_generation_response(start_datetime, response, structured_model)
        _attach_thought_signatures(parsed.content, _function_call_signatures(response))
        return parsed

    async def _parse_gemini_stream_generation_response(self, start_datetime, response, structured_model=None):  # type: ignore[override]
        signatures: list[bytes | None] = []

        async def recorded():
            async for chunk in response:
                signatures.extend(_function_call_signatures(chunk))
                yield chunk

        tool_calls: list = []
        async for parsed in super()._parse_gemini_stream_generation_response(
            start_datetime, recorded(), structured_model
        ):
            # AgentScope reports only the latest chunk's tool calls; keep all of them.
            tool_calls.extend(b for b in parsed.content if b.get("type") == "tool_use")
            _attach_thought_signatures(tool_calls, signatures)
            parsed.content = [b for b in parsed.content if b.get("type") != "tool_use"] + tool_calls
            yield parsed


class _GeminiChatFormatter(GeminiChatFormatter):
    """GeminiChatFormatter that sends each tool call's thought signature back (see _GeminiChatModel)."""

    async def _format(self, msgs: list[Msg]) -> list[dict]:  # type: ignore[override]
        formatted = await super()._format(msgs)
        signatures = [
            block.get("thought_signature")
            for msg in msgs
            for block in msg.get_content_blocks()
            if block.get("type") == "tool_use"
        ]
        parts = [part for message in formatted for part in message.get("parts", []) if "function_call" in part]
        for part, signature in zip(parts, signatures):
            if signature:
                part["thought_signature"] = base64.b64decode(signature)
        return formatted


class _AudioSafeStream:
    """Wraps an OpenAI AsyncStream so each delta's `audio` is a dict, never None."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    async def __aenter__(self) -> "_AudioSafeStream":
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._stream.__aexit__(*exc_info)

    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        async for chunk in self._stream:
            for choice in chunk.choices or []:
                if getattr(choice.delta, "audio", {}) is None:
                    choice.delta.audio = {}
            yield chunk


class _OpenAIChatModel(OpenAIChatModel):
    """OpenAIChatModel whose streaming works with current openai SDKs.

    AgentScope 1.0.7 checks `"data" in choice.delta.audio`, which raises TypeError once the SDK's
    stream deltas always have an `audio` attribute (None unless audio is requested).
    """

    def _parse_openai_stream_response(self, start_datetime, response, structured_model=None):  # type: ignore[override]
        return super()._parse_openai_stream_response(start_datetime, _AudioSafeStream(response), structured_model)


class _AnthropicChatModel(AnthropicChatModel):
    """AnthropicChatModel that keeps signed thinking blocks whose text is empty when streaming.

    Current Claude models return thinking with empty text by default (only the signature is set).
    AgentScope 1.0.7's stream parser drops such blocks, but Claude needs them passed back with the
    tool calls they precede, so we re-insert them (the non-streaming parser already keeps them).
    """

    async def _parse_anthropic_stream_completion_response(self, start_datetime, response, structured_model=None):  # type: ignore[override]
        signature = ""

        async def recorded():
            nonlocal signature
            async for event in response:
                delta = getattr(event, "delta", None)
                if event.type == "content_block_delta" and getattr(delta, "type", None) == "signature_delta":
                    signature = delta.signature
                yield event

        async for parsed in super()._parse_anthropic_stream_completion_response(
            start_datetime, recorded(), structured_model
        ):
            if signature and not any(b.get("type") == "thinking" for b in parsed.content):
                thinking_block = ThinkingBlock(type="thinking", thinking="")
                thinking_block["signature"] = signature
                parsed.content = [thinking_block, *parsed.content]
            yield parsed


# Claude models that support server-side refusal fallbacks (`fallbacks: "default"`).
_CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


def _sampling_kwargs(cfg: LLMConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        kwargs["top_p"] = cfg.top_p
    return kwargs


def _make_openai_model(cfg: LLMConfig) -> ChatModelBase:
    generate_kwargs = _sampling_kwargs(cfg)
    if cfg.max_tokens is not None:
        generate_kwargs["max_completion_tokens"] = cfg.max_tokens
    if cfg.extra_body:
        generate_kwargs["extra_body"] = cfg.extra_body
    return _OpenAIChatModel(
        model_name=cfg.model,
        api_key=cfg.api_key,
        stream=cfg.stream,
        reasoning_effort=cfg.reasoning_effort,  # type: ignore[arg-type]
        organization=cfg.organization,
        client_args={"base_url": cfg.base_url} if cfg.base_url else None,
        generate_kwargs=generate_kwargs or None,
    )


def _make_anthropic_model(cfg: LLMConfig) -> ChatModelBase:
    generate_kwargs = _sampling_kwargs(cfg)
    extra_body = dict(cfg.extra_body)
    # Claude Opus 5.5 defaults to medium effort; RTL generation and debugging warrant high.
    effort = cfg.reasoning_effort or ("high" if cfg.model == "claude-opus-5-5" else None)
    if effort:
        extra_body["output_config"] = {**extra_body.get("output_config", {}), "effort": effort}
    # Server-side refusal fallbacks are a Claude API feature; skip them behind a custom base URL.
    if cfg.refusal_fallbacks and cfg.model in _CLAUDE_FALLBACK_MODELS and not cfg.base_url:
        extra_body.setdefault("fallbacks", "default")
        generate_kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
    if extra_body:
        generate_kwargs["extra_body"] = extra_body
    return _AnthropicChatModel(
        model_name=cfg.model,
        api_key=cfg.api_key,
        max_tokens=cfg.max_tokens or 16000,
        stream=cfg.stream,
        client_args={"base_url": cfg.base_url} if cfg.base_url else None,
        generate_kwargs=generate_kwargs or None,
    )


def _make_gemini_model(cfg: LLMConfig) -> ChatModelBase:
    generate_kwargs = _sampling_kwargs(cfg)
    if cfg.max_tokens is not None:
        generate_kwargs["max_output_tokens"] = cfg.max_tokens
    if cfg.extra_body:
        generate_kwargs["http_options"] = {"extra_body": cfg.extra_body}
    return _GeminiChatModel(
        model_name=cfg.model,
        api_key=cfg.api_key,  # type: ignore[arg-type]  # None: google-genai reads GEMINI_API_KEY/GOOGLE_API_KEY
        stream=cfg.stream,
        thinking_config={"thinking_level": cfg.reasoning_effort} if cfg.reasoning_effort else None,
        client_args={"http_options": {"base_url": cfg.base_url}} if cfg.base_url else None,
        generate_kwargs=generate_kwargs or None,
    )


def make_chat_model(cfg: LLMConfig) -> ChatModelBase:
    if cfg.provider == "anthropic":
        base_model = _make_anthropic_model(cfg)
    elif cfg.provider == "gemini":
        base_model = _make_gemini_model(cfg)
    else:
        base_model = _make_openai_model(cfg)
    return UsageTrackingModel(base_model)


def make_formatter(cfg: LLMConfig | str | None = None) -> TruncatedFormatterBase:
    """Create the message formatter for the configured provider.

    A plain string is treated as an OpenAI-compatible model name (the previous signature).
    """
    if isinstance(cfg, LLMConfig):
        if cfg.provider == "anthropic":
            return AnthropicChatFormatter()
        if cfg.provider == "gemini":
            return _GeminiChatFormatter()
        model_name: str | None = cfg.model
    else:
        model_name = cfg
    if model_name and _is_llama_model(model_name):
        return Llama4ChatFormatter()
    return OpenAIChatFormatter()
