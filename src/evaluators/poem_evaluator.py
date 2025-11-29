import poetrytools
from collections import Counter
from .base_evalator import BaseEvaluator

class PoemEvaluator(BaseEvaluator):
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 512,
                 prefer_client: str = "auto"):
        """
        Initialize the poem evaluator.
        
        Args:
            model: Model name to use
            client: Optional client (defaults to auto-detect if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            prefer_client: Client preference ("auto", "ollama", or "hf")
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client)

    def rhyme_density_score(self, rhyme_scheme: str) -> float:
        if not rhyme_scheme:
            return 0.0
        
        counts = Counter(rhyme_scheme)
        n_lines = len(rhyme_scheme)
        
        participating = sum(
            1 for ch in rhyme_scheme
            if ch != 'X' and ch != ' ' and ch in "abcdefghijklmnopqrstuvwxyz" and counts[ch] >= 2  # EXCLUDE X entirely!
        )
        return participating / n_lines

    def extract_rhyme_scheme(self, poem_text: str) -> str:
        """
        Uses poetrytools to guess the rhyme scheme for the poem.
        Handles 'X' (no rhyme found) correctly in scoring.

        poetrytools.rhyme_scheme can occasionally raise IndexError (and similar)
        on odd or very short inputs, so we guard against that here and fall back
        to an empty scheme so training can continue.
        """
        try:
            tokenized = poetrytools.tokenize(poem_text)
            # poetrytools expects a list of lines; make sure we don't pass
            # obviously bad input that tends to break it.
            if not tokenized or len(tokenized) < 2:
                return ""

            scheme_list = poetrytools.rhyme_scheme(tokenized)  # e.g., ['a', 'a', 'X', 'b', 'b']
        except Exception as e:
            # Log and fall back gracefully instead of crashing the trainer
            print("[PoemEvaluator] Failed to extract rhyme scheme:", repr(e))
            print("[PoemEvaluator] Offending text (truncated to 500 chars):")
            print(poem_text[:500])
            return ""
        
        # Filter out None/empty entries if poetrytools returns them
        scheme_list = [label for label in scheme_list if label is not None and label != ""]
        
        return "".join(scheme_list)  # e.g., "aaXbb"

    def evaluate(self, base_llm_output: str, **kwargs) -> float:
        """
        Evaluates the poem output based on rhyme score.

        Args:
            base_llm_output: The output from the base LLM (poem as a string).
            **kwargs: Additional keyword arguments. May include 'gold' for comparison.

        Returns:
            A test reward score (float between 0 and 1).
        """
        # 1) Extract rhyme scheme for the generated poem
        base_llm_output = base_llm_output.replace("—", " ")
        rhyme_scheme = self.extract_rhyme_scheme(base_llm_output)

        # 2) Compute rhyme density score in [0, 1]
        rhyme_score = self.rhyme_density_score(rhyme_scheme)

        # Optionally, you could combine with other factors later, e.g.:
        # poetry_score = ...
        # final_score = 0.7 * rhyme_score + 0.3 * poetry_score


        # target = 200
        # L = len(base_llm_output.strip())
        # if L <= target:
        #     length_score = 1.0
        # else:
        #     length_score = max(0.0, 1 - (L - target) / target)

        final_score = 1.0 * rhyme_score

        return float(final_score)
