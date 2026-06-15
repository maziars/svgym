"""LLM client abstraction for SVGym — supports Anthropic and Gemini backends.

Provides a unified interface for the optimizer's tool-use loop regardless of
which LLM provider is being used.
"""

import time

import anthropic
from google import genai
from google.genai import types as genai_types

from svgym.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PROVIDER,
)


# ---------------------------------------------------------------------------
# Unified response types
# ---------------------------------------------------------------------------

class ToolCall:
    """A single tool call from the model."""
    __slots__ = ("id", "name", "args")

    def __init__(self, id: str, name: str, args: dict):
        self.id = id
        self.name = name
        self.args = args


class LLMResponse:
    """Unified response from any LLM provider."""
    __slots__ = ("tool_calls", "stop_reason", "input_tokens", "output_tokens", "raw")

    def __init__(self, tool_calls: list[ToolCall], stop_reason: str,
                 input_tokens: int, output_tokens: int, raw=None):
        self.tool_calls = tool_calls
        self.stop_reason = stop_reason  # "tool_use" or "end_turn"
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.raw = raw


# ---------------------------------------------------------------------------
# Convert our tool definitions to Gemini format
# ---------------------------------------------------------------------------

def _anthropic_tools_to_gemini(tools: list[dict]) -> list[genai_types.FunctionDeclaration]:
    """Convert Anthropic-style tool defs to Gemini FunctionDeclarations."""
    declarations = []
    for tool in tools:
        props = tool["input_schema"].get("properties", {})
        required = tool["input_schema"].get("required", [])

        # Convert properties to Gemini Schema format
        gemini_props = {}
        for pname, pdef in props.items():
            ptype = pdef.get("type", "string")
            type_map = {
                "string": "STRING",
                "integer": "INTEGER",
                "number": "NUMBER",
                "boolean": "BOOLEAN",
            }
            gemini_props[pname] = genai_types.Schema(
                type=type_map.get(ptype, "STRING"),
                description=pdef.get("description", ""),
            )

        # Build parameters schema (or None if no properties)
        parameters = None
        if gemini_props:
            parameters = genai_types.Schema(
                type="OBJECT",
                properties=gemini_props,
                required=required if required else None,
            )

        declarations.append(genai_types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=parameters,
        ))
    return declarations


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient:
    """Wrapper around the Anthropic Messages API."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY or None)
        self.model = ANTHROPIC_MODEL

    def create(self, system: str, tools: list[dict], messages: list[dict],
               max_tokens: int = 4096) -> LLMResponse:
        for attempt in range(5):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                )
                break
            except anthropic.RateLimitError:
                time.sleep(30 * (attempt + 1))
        else:
            return LLMResponse([], "end_turn", 0, 0)

        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    args=block.input,
                ))

        stop = "tool_use" if response.stop_reason == "tool_use" else "end_turn"

        return LLMResponse(
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )

    def build_assistant_message(self, response: LLMResponse) -> dict:
        """Build the assistant message to append to conversation history."""
        # Strip text blocks to save tokens — keep only tool_use blocks
        trimmed = [b for b in response.raw.content if b.type == "tool_use"]
        return {"role": "assistant", "content": trimmed or response.raw.content}

    def build_tool_results(self, results: list[dict]) -> dict:
        """Build the user message containing tool results."""
        return {"role": "user", "content": results}

    def make_tool_result(self, tool_call_id: str, content: str, is_error: bool = False,
                         tool_name: str = "") -> dict:
        result = {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        return result


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Wrapper around the Google GenAI SDK for Gemini models."""

    def __init__(self, thinking_budget: int | None = None):
        """Initialize Gemini client.

        Args:
            thinking_budget: Token budget for model thinking. None = auto (default),
                           0 = disable thinking (fastest/cheapest).
        """
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it to .env or set the environment variable. "
                "Get a key at https://aistudio.google.com/app/apikey"
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        self._chat = None
        self._tools_converted = None
        self._thinking_budget = thinking_budget

    def create(self, system: str, tools: list[dict], messages: list[dict],
               max_tokens: int = 4096) -> LLMResponse:
        # Convert tools on first call
        if self._tools_converted is None:
            self._tools_converted = _anthropic_tools_to_gemini(tools)

        # Build or reuse chat session
        if self._chat is None:
            afc_config = genai_types.AutomaticFunctionCallingConfig(
                disable=True,
            )
            config_kwargs = dict(
                system_instruction=system,
                tools=[genai_types.Tool(function_declarations=self._tools_converted)],
                max_output_tokens=max_tokens,
                automatic_function_calling=afc_config,
            )
            if self._thinking_budget is not None:
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_budget=self._thinking_budget,
                )
            self._chat = self.client.chats.create(
                model=self.model,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            # Send the initial user message
            user_text = self._extract_user_text(messages[0])
            response = self._chat.send_message(user_text)
        else:
            # Send the latest message (tool results)
            last_msg = messages[-1]
            if last_msg["role"] == "user":
                parts = self._tool_results_to_parts(last_msg)
                response = self._chat.send_message(parts)
            else:
                # Shouldn't happen, but handle gracefully
                response = self._chat.send_message("Continue.")

        # Parse response
        tool_calls = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=fc.id if hasattr(fc, 'id') and fc.id else f"gemini_{id(fc)}",
                        name=fc.name,
                        args=dict(fc.args) if fc.args else {},
                    ))

        has_tool_calls = len(tool_calls) > 0
        stop = "tool_use" if has_tool_calls else "end_turn"

        # Token usage
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        return LLMResponse(
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=response,
        )

    def _extract_user_text(self, message: dict) -> str:
        """Extract text from a user message."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Could be tool results or text blocks
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts) if texts else "Continue."
        return str(content)

    def _tool_results_to_parts(self, message: dict) -> list:
        """Convert tool result messages to Gemini FunctionResponse parts."""
        content = message.get("content", [])
        if not isinstance(content, list):
            return [str(content)]

        parts = []
        for item in content:
            if isinstance(item, dict) and "tool_call_id" in item:
                # This is our internal format — map to Gemini FunctionResponse
                parts.append(genai_types.Part.from_function_response(
                    name=item.get("name", "unknown"),
                    response={"result": item.get("content", "")},
                ))
            elif isinstance(item, dict) and item.get("type") == "tool_result":
                # Anthropic format tool_result — need the tool name
                parts.append(genai_types.Part.from_function_response(
                    name=item.get("name", "unknown"),
                    response={"result": item.get("content", "")},
                ))
        return parts if parts else ["Continue."]

    def build_assistant_message(self, response: LLMResponse) -> dict:
        """For Gemini, the chat session manages history internally."""
        return None  # Signal that no manual history management needed

    def build_tool_results(self, results: list[dict]) -> dict:
        """Build tool results message for Gemini."""
        return {"role": "user", "content": results}

    def make_tool_result(self, tool_call_id: str, content: str, is_error: bool = False,
                         tool_name: str = "") -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "name": tool_name,
            "content": content,
            "is_error": is_error,
        }

    def reset(self):
        """Reset chat session for a new SVG optimization."""
        self._chat = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(provider: str | None = None, thinking_budget: int | None = None):
    """Create an LLM client for the given provider.

    Args:
        provider: "anthropic" or "gemini". Defaults to config PROVIDER.
        thinking_budget: Gemini only — token budget for thinking.
            None = auto, 0 = disable thinking.
    """
    provider = provider or PROVIDER
    if provider == "gemini":
        return GeminiClient(thinking_budget=thinking_budget)
    return AnthropicClient()
