"""Answer generation with Qwen3."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .corpus import read_span
from .io_utils import RagError
from .models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
CONTEXT_TOKEN_BUDGET = 3000
MAX_NEW_TOKENS = 256
NO_CONTEXT_ANSWER = (
    "I could not find relevant sources to answer this question."
)
SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM codebase. "
    "Answer using only the information in the provided sources. "
    "If the sources do not contain the answer, say that the sources do not "
    "contain it instead of guessing. "
    "Answer in 2 to 5 sentences and name the relevant functions, classes, "
    "options or files when the sources mention them."
)


def load_qwen(model_name: str, dtype: torch.dtype,
              local_files_only: bool) -> tuple[Any, Any]:
    """Load the Qwen tokenizer and model.

    Args:
        model_name: Hugging Face model name.
        dtype: Weight precision.
        local_files_only: Use only the local cache.

    Returns:
        The tokenizer and the model.

    Raises:
        OSError: If the model files are not found.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, local_files_only=local_files_only
    )
    return tokenizer, model


class Generator:
    """Qwen model that answers from retrieved sources."""
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the model, on the GPU when available.

        Args:
            model_name: Hugging Face model name.

        Raises:
            RagError: If the model cannot be loaded.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            try:
                tokenizer, model = load_qwen(model_name, dtype, True)
            except OSError:
                tokenizer, model = load_qwen(model_name, dtype, False)
        except (OSError, ValueError) as error:
            raise RagError(
                f"Cannot load language model {model_name}: {error}"
            ) from None
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.model.eval()

    def build_context(self, sources: list[MinimalSource]) -> str:
        """Join source texts within the token budget.

        Args:
            sources: Sources, best first.

        Returns:
            Context text, empty if no source text.
        """
        blocks: list[str] = []
        used_tokens = 0
        for source in sources:
            text = read_span(source)
            if not text.strip():
                continue
            block = f"[Source: {source.file_path}]\n{text}"
            tokens = len(self.tokenizer.encode(block))
            if used_tokens + tokens > CONTEXT_TOKEN_BUDGET:
                break
            blocks.append(block)
            used_tokens += tokens
        return "\n\n".join(blocks)

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Generate an answer grounded in the sources.

        Args:
            question: Question to answer.
            sources: Retrieved sources, best first.

        Returns:
            The generated answer.
        """
        context = self.build_context(sources)
        if not context:
            return NO_CONTEXT_ANSWER
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return str(text).strip()
