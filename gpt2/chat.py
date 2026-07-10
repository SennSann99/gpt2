"""Chat prompt formatting shared between training and inference."""

EOS_TOKEN_ID = 50256
END_OF_TEXT = "<|endoftext|>"


def format_training_example(user_input: str, model_output: str, eos_token: str) -> str:
    return f"User: {user_input}\nModel: {model_output}{eos_token}"


def format_inference_prompt(user_input: str) -> str:
    return f"User: {user_input}\nModel: "


def format_response_prefix(user_input: str) -> str:
    """Prefix before the assistant reply; excluded from training loss."""
    return f"User: {user_input}\nModel: "


def strip_generated_response(text: str) -> str:
    if text.endswith(END_OF_TEXT):
        text = text[: -len(END_OF_TEXT)]
    return text.rstrip()
