"""
Simple NOVA food processing level classifier (template)

This script implements a rule-based, easily extensible template to classify
an entire recipe (ingredients + steps) into one of the NOVA food processing
levels 1..4 based on keyword/technique detection.

How it works (high level):
1. A dictionary of keywords/phrases is kept for each NOVA level.
2. The recipe text (ingredients + directions) is preprocessed and searched
   for those keywords using case-insensitive, whole-word regular expressions.
3. Each match increases the score for that NOVA level. We use a weighted
   scoring strategy so that detecting a higher NOVA level will generally
   outweigh many low-level matches (this matches the requirement to assign
   the final label based on the highest degree of processing found).
4. The final label is the level with the highest total score. Ties break in
   favor of the higher processing level (4 > 3 > 2 > 1).

This is a template you can extend by adding more keywords, tuning weights,
or replacing keyword matching with ML-based extraction when you have labeled
data.

Run the script to see example input/output classification.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Import centralized keyword lists and default weights
from nova_keywords import NOVA_KEYWORDS, NOVA_WEIGHTS


# ...keyword dictionaries moved to nova_keywords.py...


# ----------------------------- Utility functions -----------------------------
def build_patterns(keyword_dict: Dict[int, List[str]]) -> Dict[int, re.Pattern]:
	"""Compile regex patterns for each NOVA level.

	Each keyword/phrase is escaped and joined with |. We use word boundaries
	(\b) so we don't match substrings inside other words accidentally.
	"""
	patterns = {}
	for level, words in keyword_dict.items():
		# Sort by length descending so longer phrases are placed earlier
		# (not strictly necessary but helps regex readability).
		words_sorted = sorted(words, key=lambda s: -len(s))
		escaped = [r"(?:" + w + r")" for w in words_sorted]
		# Join into one big alternation and use word boundaries
		joined = r"\b(?:" + r"|".join(escaped) + r")\b"
		patterns[level] = re.compile(joined, flags=re.IGNORECASE)
	return patterns


def preprocess_text(text: str) -> str:
	"""Normalize recipe text to make matching robust.

	- Lowercasing is handled by regex flags, but we also replace some punctuation
	  that might break word boundaries (like slashes or commas used inside
	  phrases). We keep the text readable for debugging.
	- We do NOT aggressively remove numbers or units because they can be
	  informative later (e.g., "3 tablespoons oil").
	"""
	# Normalize whitespace and some punctuation to spaces so word boundaries
	# are consistent across lines and punctuation.
	text = text.replace("/", " ")
	text = text.replace("-", " ")
	# Collapse multiple whitespace into a single space
	text = re.sub(r"\s+", " ", text)
	return text.strip()


# ----------------------------- Core classifier -----------------------------
def classify_recipe(recipe_text: str, debug: bool = False) -> Tuple[int, Dict[int, float], Dict[int, List[str]]]:
	"""Classify the given recipe text into a NOVA level.

	Returns:
	- final_label: int (1..4)
	- scores: mapping level -> float score
	- matches: mapping level -> list of matched substrings (for explanation)

	Logic summary:
	1. Preprocess the text.
	2. For each NOVA level, find all non-overlapping regex matches.
	3. Each match increases that level's score by the level weight.
	4. Final label = argmax over (level, score). If all scores are zero,
	   default to 1 (minimally processed / whole-food style).
	5. If tie in score, choose the higher processing level (max(level)).
	"""
	if not recipe_text or not recipe_text.strip():
		raise ValueError("Empty recipe text provided")

	text = preprocess_text(recipe_text)
	patterns = build_patterns(NOVA_KEYWORDS)

	scores: Dict[int, float] = defaultdict(float)
	matches: Dict[int, List[str]] = defaultdict(list)

	for level, pattern in patterns.items():
		# finditer returns match objects; we capture the matched text for explainability
		for m in pattern.finditer(text):
			matched = m.group(0)
			matches[level].append(matched)
			scores[level] += NOVA_WEIGHTS.get(level, 1.0)

	# Debug printing if requested
	if debug:
		print("DEBUG: raw matches per level:")
		for lvl in sorted(matches.keys()):
			print(f"  Level {lvl}: {matches[lvl]} (score {scores[lvl]})")

	# Determine final label. If no matches at all, default to 1.
	if not scores:
		final_label = 1
	else:
		# Pick the level with highest score; ties -> higher level
		# Build list of (score, level) and pick max by (score, level)
		best = max(((s, lvl) for lvl, s in scores.items()), key=lambda t: (t[0], t[1]))
		final_label = best[1]

	return final_label, dict(scores), dict(matches)


# ----------------------------- Example usage -----------------------------
EXAMPLES = [
	(
		"Simple boiled vegetable",
		"Ingredients:\n- Carrots\n- Potato\n\nDirections:\nBoil the carrots and potatoes until tender. Drain and serve.",
	),
	(
		"Ground/refined ingredient example",
		"Ingredients:\n- Wheat flour (refined)\n- Water\n\nDirections:\nSift and grind the flour; use the refined flour to make dough.",
	),
	(
		"Added sugar + frying",
		"Ingredients:\n- Sugar\n- Oil\n\nDirections:\nAdd sugar, heat oil, fry the batter until golden.",
	),
	(
		"Highly processed instant mix",
		"Ingredients:\n- Instant mix powder, artificial flavors, preservatives\n\nDirections:\nMix with water and serve. Contains preservatives and artificial coloring.",
	),
]


def _run_examples():
	print("NOVA classifier examples:\n")
	for title, text in EXAMPLES:
		label, scores, matches = classify_recipe(text)
		print(f"Example: {title}")
		print(f"  Predicted NOVA level: {label}")
		print(f"  Scores: {scores}")
		print(f"  Matches: {matches}")
		print()


if __name__ == "__main__":
	# Run examples as a quick smoke test. In production you would import
	# classify_recipe() and apply it to your dataset.
	_run_examples()

