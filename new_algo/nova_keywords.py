"""
Expanded NOVA keyword dictionaries and default weights.

This module centralizes NOVA keyword/phrase lists (both ingredient- and step-like terms)
for each NOVA level, including cosmetic additive families, E-number/INS patterns,
and industrial processing markers.

Notes:
- Broad families (e.g., emulsifier, flavour enhancer) plus common members are included.
- E-number and INS patterns are provided as regex-like strings; ensure the matcher
  supports regex OR replace with an explicit allowlist.
- Ambiguous verbs remain but should be kept at lower weights in the classifier.
"""

from typing import Dict, List

# Core NOVA keyword lists by level (1..4).
# Include both ingredient-like tokens and step/verb tokens since this module is shared.
NOVA_KEYWORDS: Dict[int, List[str]] = {
    1: [
        # States and whole/minimally processed descriptors
        "raw", "fresh", "uncooked", "whole food", "wholegrain", "whole grain", "plain",
        # Basic prep / cleaning
        "wash", "washing", "rinse", "rinsed", "soak", "peel", "peeling",
        "slice", "sliced", "chop", "chopped", "dice", "diced", "mince", "minced",
        "grate", "grated", "shred", "shredded",
        # Minimal cooking/heat
        "boil", "boiling", "simmer", "steam", "steaming", "blanch", "blanched",
        "poach", "poached",
        # Cold storage / minimal preservation
        "freeze", "freezing", "frozen", "chill", "refrigerate", "refrigerated", "thaw",
        # Home-style minimal processing
        "cook", "cooked", "stew", "stewed", "roast", "roasted", "toast", "toasted",
        # Simple fermentation and culturing cues (handle context carefully)
        "ferment", "fermented", "cultured", "sprouted",
    ],

    2: [
        # Industrial derivation verbs for culinary ingredients
        "refine", "refined", "refining", "refinement",
        "grind", "grinding", "ground", "mill", "milling",
        "extract", "extracting", "extracted",
        "press", "pressed", "cold press", "cold-pressed", "pressing",
        "clarify", "clarified", "deodorize", "deodorized", "bleached",
        "centrifuge", "fractionate", "fractionated", "rendered",
        "dry", "drying", "dehydrate", "dehydrated", "powder", "powdered",
        "sift", "sifted", "sieving", "filter", "filtered",
        # Processed culinary ingredients
        "vegetable oil", "refined oil", "olive oil", "canola oil", "sunflower oil", "palm oil",
        "rice bran oil", "mustard oil", "groundnut oil", "sesame oil",
        "white flour", "refined flour", "all-purpose flour", "maida", "cornstarch", "starch",
        "wheat flour", "rice flour",
        "sugar", "granulated sugar", "caster sugar", "powdered sugar", "icing sugar",
        "sucrose",
        "salt", "table salt", "sodium chloride",
        "butter", "ghee", "lard",
        "honey", "maple syrup", "jaggery",
    ],

    3: [
        # Additions that transform foods for durability/palatability
        "add salt", "adding salt", "salted",
        "add sugar", "adding sugar", "sweetened",
        "add oil", "adding oil", "in oil",
        "vinegar", "soy sauce", "ketchup", "mustard", "brine", "in brine", "in syrup",
        # Processing/preservation methods
        "bake", "baking", "baked",
        "fry", "frying", "fried", "pan-fry", "deep-fry", "deep fry", "shallow fry", "air fry",
        "grill", "grilled", "roast", "roasted", "smoke", "smoked",
        "pickle", "pickling", "pickled",
        "cure", "curing", "cured",
        "can", "canned", "tinned", "bottle", "bottled", "jarred",
        "preserve", "preserving", "preserved",
        "jam", "jelly", "marmalade",
        "pasteurize", "pasteurized", "pasteurise", "pasteurised",
        # Processed foods without cosmetic additives
        "processed cheese", "cheese spread",
        "canned vegetables", "canned fish", "canned beans",
        "salt-cured", "sugar-cured",
        "mass-produced bread", "packaged bread",
    ],

    4: [
        # Cosmetic additive families (generic)
        "preservative", "preservatives", "antioxidant", "antioxidants",
        "emulsifier", "emulsifiers", "stabilizer", "stabiliser", "stabilizers", "stabilisers",
        "thickener", "thickeners", "gelling agent", "gelling agents",
        "flavor", "flavour", "flavoring", "flavouring", "flavorings", "flavourings",
        "color", "colour", "coloring", "colouring", "artificial color", "artificial colour",
        "natural flavor", "natural flavour", "flavor enhancer", "flavour enhancer",
        "sweetener", "sweeteners",
        # Specific flavor enhancers and related
        "monosodium glutamate", "msg",
        "disodium inosinate", "inosinate", "e631",
        "disodium guanylate", "guanylate", "e627",
        "yeast extract", "autolyzed yeast", "autolysed yeast",
        "hydrolyzed vegetable protein", "hydrolyzed protein", "hydrolysed protein",
        # Preservatives
        "sodium benzoate", "potassium benzoate", "benzoate",
        "potassium sorbate", "sorbate",
        "sodium nitrite", "nitrite",
        "sodium nitrate", "nitrate",
        "sodium metabisulfite", "sodium metabisulphite", "sulphite", "sulfite",
        "calcium propionate", "propionate",
        "bht", "bha", "tbhq", "edta",
        # Emulsifiers/thickeners/stabilizers (common examples)
        "lecithin", "soy lecithin", "sunflower lecithin",
        "mono- and diglycerides", "mono and diglycerides", "e471",
        "datem", "diacetyl tartaric acid ester of mono- and diglycerides",
        "ssl", "sodium stearoyl lactylate", "csl", "calcium stearoyl lactylate",
        "polysorbate 60", "polysorbate 80",
        "propyl gallate",
        "propylene glycol monoesters", "pgme",
        "polyglycerol esters", "pge",
        "carboxymethylcellulose", "cmc", "cellulose gum",
        "xanthan gum", "guar gum", "gum arabic", "arabic gum",
        "carrageenan", "agar", "locust bean gum", "methylcellulose",
        "modified starch", "modified corn starch", "modified wheat starch",
        # Sweeteners (non-nutritive and polyols)
        "aspartame", "sucralose", "acesulfame k", "acesulfame-k", "saccharin", "cyclamate",
        "stevia extract", "rebaudioside a", "reb a", "neotame", "advantame",
        "erythritol", "xylitol", "sorbitol", "mannitol", "isomalt", "maltitol", "lactitol",
        # Sugars/syrups strongly linked to industrial formulations
        "high fructose corn syrup", "hfcs", "glucose syrup", "corn syrup", "corn syrup solids",
        "invert sugar", "maltodextrin", "dextrose",
        # Protein isolates/structured proteins
        "textured vegetable protein", "tvp", "soy protein isolate", "whey protein isolate",
        "protein isolate", "hydrolyzed collagen", "gelatin hydrolysate",
        # Industrial/packaged product markers
        "instant mix", "instant", "premix", "seasoning packet", "seasoning sachet",
        "flavor sachet", "powdered soup", "soup powder", "sauce powder",
        "instant noodles", "instant soup",
        "meal replacement", "nutritional shake",
        "ready to eat", "ready-to-eat", "ready meal", "ready-made", "prepackaged", "pre-packaged",
        "shelf stable", "shelf-stable", "long shelf life",
        "reconstituted", "reformed", "formed", "restructured",
        # Industrial processing techniques
        "extruded", "extrusion cooked", "extrusion-cooked", "uht", "ultra-heat treated",
        "spray-dried", "spray dried", "agglomerated", "enzyme-treated",
        "hydrogenated", "partially hydrogenated", "interesterified",
    ],
}

# Optional regex patterns (store as strings; ensure classifier treats them as regex)
# Families for broad capture, plus INS codes used in India and other markets.
NOVA_REGEX_PATTERNS: Dict[int, List[str]] = {
    4: [
        # Generic E-number capture (E100–E1599). Use case-insensitive regex in matcher.
        r"\bE[1-9]\d{2,3}\b",
        # Family-specific ranges (helpful if you support family mapping)
        r"\bE1\d{2}\b",   # colours (E100–E199)
        r"\bE2\d{2}\b",   # preservatives (E200–E299)
        r"\bE3\d{2}\b",   # antioxidants / acidity regulators (E300–E399)
        r"\bE4\d{2}\b",   # thickeners / stabilisers / emulsifiers (E400–E499)
        r"\bE6\d{2}\b",   # flavour enhancers (E600–E699)
        r"\bE9\d{2}\b",   # glazing agents / gases / sweeteners (E900–E999)
        # INS codes (often appear without 'E')
        r"\bINS\W?\d{3,4}\b",
    ],
}

# Default weights per NOVA level (tune during calibration).
NOVA_WEIGHTS: Dict[int, float] = {
    1: 1.0,
    2: 1.5,
    3: 2.0,
    4: 3.0,
}

if __name__ == "__main__":
    print("NOVA_KEYWORDS summary:")
    for lvl in sorted(NOVA_KEYWORDS.keys()):
        print(f"Level {lvl}: {len(NOVA_KEYWORDS[lvl])} phrases, weight {NOVA_WEIGHTS[lvl]}")
    print("Regex patterns for NOVA 2:", len(NOVA_REGEX_PATTERNS.get(2, [])))
