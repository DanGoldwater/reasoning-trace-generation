"""The wording put to the model, kept in one place so runs stay comparable."""

from src.dataset.models import HFSample

ANSWER_INSTRUCTIONS = (
    "Answer the multiple-choice question using your existing knowledge. "
    "Keep your reasoning brief: make one assessment, then immediately give "
    "your final answer using exactly one offered option key. "
    "If uncertain, acknowledge uncertainty briefly and choose the most likely "
    "option. Do not repeatedly reconsider your decision. "
    "You have no search tools or database access: do not simulate searches, "
    "invent citations, or claim to have verified experimental results. "
    "Do not repeat the question or draft multiple answers."
)


def render_question(sample: HFSample) -> str:
    """Lay a question and its options out for the model, keys in a stable order."""
    options = "\n".join(
        f"{key}: {sample.options[key]}" for key in sorted(sample.options)
    )
    return f"{sample.question}\n{options}"
