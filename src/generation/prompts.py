"""The wording put to the model, kept in one place so runs stay comparable."""

from src.dataset.models import HFSample

ANSWER_INSTRUCTIONS = (
    "You are answering a multiple-choice question about cancer biology and drug "
    "response. Reason it through, then commit to exactly one option and answer "
    "with that option's key."
)


def render_question(sample: HFSample) -> str:
    """Lay a question and its options out for the model, keys in a stable order."""
    options = "\n".join(
        f"{key}: {sample.options[key]}" for key in sorted(sample.options)
    )
    return f"{sample.question}\n{options}"
