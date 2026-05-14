# Vocabulary list for CLIP along with descriptive sentences for better 
# identification of mood 
MOODS = {
    "happy_upbeat":    "a vibrant joyful photo full of energy and sunshine",
    "chill_relaxed":   "a calm peaceful photo with a relaxed laid-back vibe",
    "romantic_dreamy": "a soft romantic photo with a dreamy aesthetic",
    "moody_dark":      "a dark moody photo with a mysterious atmosphere",
    "nostalgic":       "a nostalgic vintage-feeling photo with warm tones",
    "energetic_party": "an exciting party photo with movement and crowds",
    "ethereal":        "an ethereal magical photo that feels otherworldly",
    "cozy_intimate":   "a cozy intimate photo with warm soft lighting",
    "adventurous":     "an adventurous outdoor photo with epic landscapes",
    "edgy_cool":       "a stylish edgy photo with a cool urban vibe",
    "melancholic":     "a quiet melancholic photo with a wistful feeling",
    "fun_playful":     "a fun playful photo with bright cheerful colors",
}
 
# Summarising each mood with related "tags" - Last.fm relies on tags to
# pool related tracks.
MOOD_TO_LASTFM_TAGS = {
    "happy_upbeat":    ["happy", "feel good", "upbeat"],
    "chill_relaxed":   ["chillout", "chill", "relaxing"],
    "romantic_dreamy": ["romantic", "dream pop", "love songs"],
    "moody_dark":      ["dark", "moody", "atmospheric"],
    "nostalgic":       ["nostalgia", "80s", "oldies"],
    "energetic_party": ["party", "dance", "club"],
    "ethereal":        ["ethereal", "dream pop", "ambient"],
    "cozy_intimate":   ["acoustic", "singer-songwriter", "mellow"],
    "adventurous":     ["indie folk", "epic", "uplifting"],
    "edgy_cool":       ["indie rock", "alternative", "garage rock"],
    "melancholic":     ["melancholy", "sad", "bittersweet"],
    "fun_playful":     ["fun", "indie pop", "feel good"],
}

# Sanity check: every mood must appear in both dictionaries. This runs once
# when the module is imported and catches typos before they cause confusing
# bugs deep in the pipeline.
assert set(MOODS.keys()) == set(MOOD_TO_LASTFM_TAGS.keys()), (
    "MOODS and MOOD_TO_LASTFM_TAGS must have the same keys. "
    f"Mismatch: {set(MOODS.keys()) ^ set(MOOD_TO_LASTFM_TAGS.keys())}"
)