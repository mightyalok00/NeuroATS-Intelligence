"""
Text Preprocessor and Stylistic Feature Extraction Pipeline
Analyzes readability, action verb density, quantifiable metrics, and bullet strength.
"""

import re

from src.config import ALL_ACTION_VERBS, POWER_ACTION_VERBS


class ResumePreprocessor:
    """Computes NLP, lexical, and stylistic features from processed text and bullet points."""

    # Quantifiable metrics regex pattern (numbers, percentages, dollars, multipliers)
    METRIC_PATTERN = re.compile(
        r"(?:\b\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?(?:[kKmMbB]|(?:\s*million|\s*thousand|\s*billion))?|\b\d+(?:\.\d+)?(?:x|X)\b|\b\d{1,3}(?:,\d{3})+\b|\b\d+\+?(?:\s*(?:users|clients|customers|requests|qps|rps|tps|ms|seconds|minutes|hours|days|weeks|months|years|gb|tb|pb))\b|\b(?:reduced|increased|improved|boosted|saved|grew)\b[^.]{0,35}\b\d+)",
        re.IGNORECASE
    )

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Basic text cleaning: remove extra whitespace and non-standard characters."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def tokenize_words(cls, text: str) -> list[str]:
        """Simple word tokenizer without punctuation."""
        return re.findall(r"\b[a-zA-Z0-9_\-+#.]+\b", text.lower())

    @classmethod
    def calculate_readability(cls, text: str) -> dict[str, float]:
        """
        Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.
        Formula:
        Flesch Reading Ease = 206.835 - 1.015*(total words/total sentences) - 84.6*(total syllables/total words)
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) >= 3]
        words = cls.tokenize_words(text)

        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)

        # Approximate syllable counter
        def count_syllables(word: str) -> int:
            word = word.lower()
            count = len(re.findall(r"[aeiouy]+", word))
            if word.endswith("e") and not word.endswith("le") and count > 1:
                count -= 1
            return max(count, 1)

        total_syllables = sum(count_syllables(w) for w in words)

        words_per_sentence = num_words / num_sentences
        syllables_per_word = total_syllables / num_words

        flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        flesch_reading_ease = max(0.0, min(100.0, flesch_reading_ease))

        grade_level = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
        grade_level = max(1.0, min(20.0, grade_level))

        # Lexical diversity: Type-Token Ratio (TTR)
        unique_words = len(set(words))
        lexical_diversity = (unique_words / num_words) if num_words > 0 else 0.0

        return {
            "total_words": num_words,
            "total_sentences": num_sentences,
            "words_per_sentence": round(words_per_sentence, 1),
            "flesch_reading_ease": round(flesch_reading_ease, 1),
            "grade_level": round(grade_level, 1),
            "lexical_diversity": round(lexical_diversity * 100, 1),
        }

    @classmethod
    def analyze_bullets(cls, bullets: list[str]) -> dict[str, any]:
        """
        Audit bullet points for:
        1. Starting with power action verbs
        2. Containing quantifiable metrics
        3. Word count distribution
        """
        if not bullets:
            return {
                "total_bullets": 0,
                "action_verb_count": 0,
                "action_verb_ratio": 0.0,
                "quantifiable_count": 0,
                "quantifiable_ratio": 0.0,
                "verb_category_breakdown": {},
                "bullet_details": []
            }

        action_verb_count = 0
        quantifiable_count = 0
        verb_category_breakdown: dict[str, int] = {k: 0 for k in POWER_ACTION_VERBS}
        bullet_details: list[dict] = []

        for bullet in bullets:
            words = bullet.split()
            first_word = words[0].lower().strip(".,:;!?") if words else ""
            
            # Check if starts with or contains action verb
            has_action_verb = first_word in ALL_ACTION_VERBS
            detected_verb = first_word if has_action_verb else None

            if not has_action_verb:
                # Check within first 3 words
                for w in words[:3]:
                    clean_w = w.lower().strip(".,:;!?")
                    if clean_w in ALL_ACTION_VERBS:
                        has_action_verb = True
                        detected_verb = clean_w
                        break

            if has_action_verb:
                action_verb_count += 1
                # Find category
                for cat, verbs in POWER_ACTION_VERBS.items():
                    if detected_verb in verbs:
                        verb_category_breakdown[cat] += 1
                        break

            # Check quantifiable metrics
            has_metric = bool(cls.METRIC_PATTERN.search(bullet))
            if has_metric:
                quantifiable_count += 1

            strength_score = 0
            if has_action_verb:
                strength_score += 40
            if has_metric:
                strength_score += 40
            if 10 <= len(words) <= 25:
                strength_score += 20

            bullet_details.append({
                "bullet": bullet,
                "has_action_verb": has_action_verb,
                "detected_verb": detected_verb,
                "has_metric": has_metric,
                "word_count": len(words),
                "strength_score": strength_score
            })

        total = len(bullets)
        return {
            "total_bullets": total,
            "action_verb_count": action_verb_count,
            "action_verb_ratio": round((action_verb_count / total) * 100, 1) if total > 0 else 0.0,
            "quantifiable_count": quantifiable_count,
            "quantifiable_ratio": round((quantifiable_count / total) * 100, 1) if total > 0 else 0.0,
            "verb_category_breakdown": verb_category_breakdown,
            "bullet_details": bullet_details
        }
