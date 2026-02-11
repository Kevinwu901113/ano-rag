from relrag.utils.text_utils import TextUtils
from relrag.utils.token_counter import TokenCounter


def test_token_counter_rough_vs_real():
    text = (
        "Question: Which city hosted the event?\n"
        "Evidence: The ceremony was held at the Tropicana Hotel and Casino in Paradise, Nevada.\n"
        "Please answer with only the canonical label."
    )
    messages = [
        {"role": "system", "content": "You are a factual answerer."},
        {"role": "user", "content": text},
    ]

    rough = TextUtils.rough_token_len(text)
    real_text = TokenCounter.count_text(text)
    real_messages = TokenCounter.count_messages(messages)

    # Keep this printable for quick manual verification with `pytest -s`.
    print(f"token_counter mode={'real' if TokenCounter._load_tokenizer() is not None else 'fallback'}")
    print(f"rough={rough} real_text={real_text} real_messages={real_messages}")

    assert isinstance(real_text, int)
    assert isinstance(real_messages, int)
    assert real_text > 0
    assert real_messages >= real_text
