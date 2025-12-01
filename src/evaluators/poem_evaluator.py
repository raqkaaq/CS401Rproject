import poetrytools
from collections import Counter
import re
from .base_evalator import BaseEvaluator

class PoemEvaluator(BaseEvaluator):
    def __init__(self, model: str, client=None, temperature: float = 0.0, max_tokens: int = 512,
                 prefer_client: str = "auto", evaluator_8bit: bool = False):
        """
        Initialize the poem evaluator.
        
        Args:
            model: Model name to use
            client: Optional client (defaults to auto-detect if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            prefer_client: Client preference ("auto", "ollama", or "hf")
            evaluator_8bit: If True, use 8-bit quantization for HFClient (reduces memory usage)
        """
        super().__init__(model, client, temperature, max_tokens, prefer_client=prefer_client, evaluator_8bit=evaluator_8bit)

    def rhyme_density_score(self, rhyme_scheme: str) -> float:
        if not rhyme_scheme:
            return 0.0
        
        counts = Counter(rhyme_scheme)
        n_lines = len(rhyme_scheme)
        
        participating = sum(
            1 for ch in rhyme_scheme
            if ch != 'X' and ch != ' ' and ch in "abcdefghijklmnopqrstuvwxyz" and counts[ch] >= 2  # EXCLUDE X entirely!
        )
        return participating / n_lines if n_lines > 0 else 0.0
    
    def repetition_penalty(self, poem_text: str) -> float:
        """
        Penalizes excessive repetition of lines or phrases.
        Returns a score between 0 and 1, where 1 means no repetition penalty.
        """
        lines = [line.strip() for line in poem_text.strip().split('\n') if line.strip()]
        if len(lines) < 2:
            return 1.0
        
        # Count occurrences of each line
        line_counts = Counter(lines)
        
        # Calculate repetition ratio: unique lines / total lines
        unique_ratio = len(line_counts) / len(lines)
        
        # Also check for very high frequency of any single line
        max_freq = max(line_counts.values()) if line_counts else 1
        max_freq_penalty = 1.0 - min(0.5, (max_freq - 1) / len(lines))
        
        # Combine both penalties
        return min(unique_ratio, max_freq_penalty)
    
    def diversity_score(self, poem_text: str) -> float:
        """
        Rewards lexical diversity (unique words relative to total words).
        Returns a score between 0 and 1.
        """
        # Extract words (lowercase, alphanumeric only)
        words = re.findall(r'\b[a-z]+\b', poem_text.lower())
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Diversity ratio
        diversity = unique_words / total_words
        
        # Also reward having a reasonable vocabulary size
        vocab_bonus = min(1.0, unique_words / 20.0)  # Bonus for having at least 20 unique words
        
        return 0.7 * diversity + 0.3 * vocab_bonus
    
    def length_score(self, poem_text: str, target_length: int = 200, tolerance: int = 100) -> float:
        """
        Rewards poems of appropriate length.
        Returns a score between 0 and 1.
        """
        length = len(poem_text.strip())
        
        if length == 0:
            return 0.0
        
        # Ideal range: target_length ± tolerance
        if abs(length - target_length) <= tolerance:
            return 1.0
        elif length < target_length - tolerance:
            # Too short
            return max(0.0, length / (target_length - tolerance))
        else:
            # Too long - exponential decay
            excess = length - (target_length + tolerance)
            return max(0.0, 1.0 - (excess / (target_length + tolerance)))
    
    def semantic_coherence_score(self, poem_text: str) -> float:
        """
        Basic check for semantic coherence - penalizes obvious repetition patterns
        and rewards having varied sentence structures.
        """
        lines = [line.strip() for line in poem_text.strip().split('\n') if line.strip()]
        if len(lines) < 2:
            return 0.5  # Neutral for very short poems
        
        # Check if all lines start the same way (strong indicator of repetition)
        line_starts = [line.split()[0].lower() if line.split() else "" for line in lines[:10]]  # Check first 10
        unique_starts = len(set(line_starts))
        start_diversity = unique_starts / len(line_starts) if line_starts else 0.0
        
        # Check average line length variation
        line_lengths = [len(line) for line in lines]
        if len(line_lengths) > 1:
            avg_length = sum(line_lengths) / len(line_lengths)
            variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)
            # Normalize variance (higher variance = more variety = better)
            length_variety = min(1.0, variance / 100.0)  # Rough normalization
        else:
            length_variety = 0.5
        
        return 0.6 * start_diversity + 0.4 * length_variety

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
        Evaluates the poem output based on multiple factors to prevent reward hacking.
        Combines rhyme, diversity, repetition penalty, length, and semantic coherence.

        Args:
            base_llm_output: The output from the base LLM (poem as a string).
            **kwargs: Additional keyword arguments. May include 'gold' for comparison.

        Returns:
            A test reward score (float between 0 and 1).
        """
        # Clean the input
        base_llm_output = base_llm_output.replace("—", " ")
        
        # 1) Rhyme score (weighted lower to prevent over-optimization)
        rhyme_scheme = self.extract_rhyme_scheme(base_llm_output)
        rhyme_score = self.rhyme_density_score(rhyme_scheme)
        
        # 2) Repetition penalty (critical to prevent "The endless" loops)
        repetition_score = self.repetition_penalty(base_llm_output)
        
        # 3) Diversity score (rewards varied vocabulary)
        diversity_score = self.diversity_score(base_llm_output)
        
        # 4) Length score (rewards appropriate poem length)
        length_score = self.length_score(base_llm_output, target_length=200, tolerance=100)
        
        # 5) Semantic coherence (rewards varied structure)
        coherence_score = self.semantic_coherence_score(base_llm_output)
        
        # Combine scores with weights
        # Repetition penalty is critical - apply it multiplicatively to prevent reward hacking
        base_score = (
            0.25 * rhyme_score +           # Rhyme is important but not everything
            0.25 * diversity_score +       # Diversity prevents repetition
            0.20 * length_score +          # Appropriate length
            0.30 * coherence_score         # Coherence prevents degenerate patterns
        )
        
        # Apply repetition penalty multiplicatively (strong penalty for repetition)
        # This ensures that even if other scores are high, repetition kills the reward
        final_score = base_score * (0.3 + 0.7 * repetition_score)
        
        return float(max(0.0, min(1.0, final_score)))
