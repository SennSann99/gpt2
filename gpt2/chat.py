"""Chat prompt formatting shared between training and inference."""

from collections.abc import Sequence

EOS_TOKEN_ID = 50256
END_OF_TEXT = "<|endoftext|>"
ROLE_PREFIXES = {
    "system": "System: ",
    "user": "User: ",
    "assistant": "Model: ",
}


def format_chat_message(role: str, content: str, eos_token: str = END_OF_TEXT) -> str:
    """Format one Smol-SmolTalk message using this model's chat template."""
    if role not in ROLE_PREFIXES:
        raise ValueError(f"Unsupported chat role: {role}")
    suffix = eos_token if role == "assistant" else "\n"
    return f"{ROLE_PREFIXES[role]}{content}{suffix}"


def format_conversation(
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
    eos_token: str = END_OF_TEXT,
) -> str:
    """Format a complete multi-turn conversation."""
    text = "".join(
        format_chat_message(message["role"], message["content"], eos_token)
        for message in messages
    )
    if add_generation_prompt:
        text += ROLE_PREFIXES["assistant"]
    return text


def format_training_example(user_input: str, model_output: str, eos_token: str) -> str:
    return format_conversation(
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": model_output},
        ],
        eos_token=eos_token,
    )


def format_inference_prompt(messages: Sequence[dict[str, str]] | str) -> str:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return format_conversation(messages, add_generation_prompt=True)


def format_response_prefix(user_input: str) -> str:
    """Prefix before the assistant reply; excluded from training loss."""
    return f"User: {user_input}\nModel: "


def strip_generated_response(text: str) -> str:
    if text.endswith(END_OF_TEXT):
        text = text[: -len(END_OF_TEXT)]
    return text.rstrip()
