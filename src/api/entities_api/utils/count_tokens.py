from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def _get_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name)  # nosec B615


def count_tokens(input_string: str, tokenizer_name: str = "gpt2") -> int:
    try:
        tokenizer = _get_tokenizer(tokenizer_name)
        tokens = tokenizer.encode(input_string, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        raise Exception(
            f"Failed to load or use the tokenizer '{tokenizer_name}': {str(e)}"
        )
