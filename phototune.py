"""
PhotoTune: Passes a photo input into neural networks paired with free
music APIs to recommend top songs 
 
Pipeline:
  1. BLIP captions the photo (transformer-based vision-language neural net)
  2. CLIP scores the photo against mood descriptions (transformer neural net)
  3. Last.fm fetches popular tracks for the top mood tags
  4. Deezer search enriches each track with a playable URL + album art + preview
 
Usage:
    python phototune.py path/to/photo.jpg
    python phototune.py photo.jpg --moods 2 --songs 6
"""
import argparse
import os
import random
import sys
from pathlib import Path
 
import requests
import torch
from PIL import Image
import pillow_avif  # adds AVIF support to PIL
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
)

from moods import MOODS, MOOD_TO_LASTFM_TAGS

from dotenv import load_dotenv
load_dotenv()

 
"""
JSON response format 
{
    "tracks": {
        "track": [
            {
                "name": "...", 
                "artist": {"name": "...", "mbid": "..." },
                "playcount": "...", 
                "url": "..."
            },
            ...
        ]
    }
}
"""
# Last.fm object - returns top songs based on mood tags
class LastFM:
    # Last.fm's API URL 
    BASE = "https://ws.audioscrobbler.com/2.0/"
 
    def __init__(self, api_key: str):
        self.api_key = api_key
 
    def top_tracks_for_tags(self, tags: list[str], pool_size: int = 100) -> list[tuple[str, str]]:
        """Pool top tracks across multiple tags and returns list of unique (title, artist) pairs."""
        seen_songs, pooled_songs = set(), []
        songs_per_tag = max(20, pool_size // len(tags))
 
        for tag in tags:
            r = requests.get(self.BASE, timeout=10, params={
                "method": "tag.gettoptracks",
                "tag": tag,
                "api_key": self.api_key,
                "format": "json",
                "limit": songs_per_tag,
            })

            # If request failed, skip this tag - otherwise parse the JSON response
            if not r.ok:
                continue
            tracks = r.json().get("tracks", {}).get("track", [])
            for track in tracks:
                key = (track["name"].lower(), track["artist"]["name"].lower())
                if key in seen_songs:
                    continue
                seen_songs.add(key)
                pooled_songs.append((track["name"], track["artist"]["name"]))

        return pooled_songs
 
# Deezer object - takes (title, artist) lists and finds corresponding song urls etc 
class Deezer:
    # Deezer's API URL
    BASE = "https://api.deezer.com"
 
    def find_track(self, title: str, artist: str) -> dict | None:
        """Find the canonical Deezer entry for a (title, artist) pair."""
        # Deezer's advanced search lets us pin down by both fields precisely.
        query = f'track:"{title}" artist:"{artist}"'
        r = requests.get(
            f"{self.BASE}/search",
            params={"q": query, "limit": 1},
            timeout=10,
        )
        if not r.ok:
            return None
        items = r.json().get("data", [])
        if not items:
            return None
        t = items[0]
        return {
            "title": t["title"],
            "artist": t["artist"]["name"],
            "deezer_url": t["link"],
            "preview_url": t.get("preview"),  
            "album_art": t["album"].get("cover_medium") or t["album"].get("cover"),
            "duration": t.get("duration", 0),
        }
 
# Creating Phototune AI pipeline 
class PhotoTune:
    def __init__(self, lastfm: LastFM, deezer: Deezer, device: str | None = None):
        self.lastfm = lastfm
        self.deezer = deezer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading neural networks on {self.device}... (first run downloads ~1GB)")
 
        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
 
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
 
        self.mood_ids = list(MOODS.keys())
        self.mood_prompts = list(MOODS.values())
 
    """
    Image -> inputs (via processor)
    Inputs -> ids (via the model)
    Ids -> text (via processor decoding)
    """
    def caption(self, image: Image.Image) -> str:
        # Converts image into grid of numbers BLIP can understand
        inputs = self.blip_proc(image, return_tensors="pt").to(self.device)

        # Takes the numbers and produces numbers (BLIP's internal code) for a word (e.g. 1037 = 'a')
        with torch.no_grad():
            ids = self.blip.generate(**inputs, max_new_tokens=40)

        # Converts numbers (caption tokens) into human-readable text
        return self.blip_proc.decode(ids[0], skip_special_tokens=True)
 
    # Returns a ranked lists of moods from highest probability to lowest
    """
    e.g.
    [
        ("romantic_dreamy", 0.31), 
        ("nostalgic", 0.24),
        ("ethereal", 0.18),
        ...
    ]
    """
    def score_moods(self, image: Image.Image) -> list[tuple[str, float]]:
        # Convert image and mood prompts into tensors CLIP can process.
        # - text/images: the 12 prompts get tokenized and the image gets resized/normalized
        # - return_tensors="pt": output as PyTorch tensors
        # - padding=True: pad shorter prompts to match the longest so they form a clean batch
        """
        e.g. inputs = 
        {
            "pixel_values": tensor of shape [1, 3, 224, 224],   
            "input_ids": tensor of shape [12, 12],
            "attention_mask": tensor of shape [12, 12],
        }
        """
        inputs = self.clip_proc(
            text=self.mood_prompts, images=image,
            return_tensors="pt", padding=True,
        ).to(self.device)

        with torch.no_grad():
            out = self.clip(**inputs)

        # Convert CLIP's raw similarity scores into probabilities that sum to 1,
        # then strip the batch dimension and move to a plain NumPy array.
        probs = out.logits_per_image.softmax(dim=-1).squeeze().cpu().numpy()

        # Pair each mood ID with its probability, then sort highest-to-lowest.
        ranked = sorted(zip(self.mood_ids, probs), key=lambda kv: kv[1], reverse=True)
        return [(m, float(p)) for m, p in ranked]
 
    def recommend(self, image_path: str, n_moods: int = 2, songs_per_mood: int = 5) -> dict:
        image = Image.open(image_path).convert("RGB")
        caption = self.caption(image)
        moods = self.score_moods(image)
 
        # Build recs mood-by-mood. For each mood, pull a candidate pool from
        # Last.fm, sample randomly so repeated runs feel fresh, then enrich
        # via Deezer so the user gets a clickable link + preview audio.
        all_recs = []
        for mood_id, score in moods[:n_moods]:
            tags = MOOD_TO_LASTFM_TAGS[mood_id]
            candidates = self.lastfm.top_tracks_for_tags(tags, pool_size=50)
            random.shuffle(candidates)
 
            mood_recs = []
            for title, artist in candidates:
                if len(mood_recs) >= songs_per_mood:
                    break
                deezer_data = self.deezer.find_track(title, artist)
                if deezer_data is None:
                    continue  # Skip tracks Deezer doesn't have.
                mood_recs.append({
                    **deezer_data,
                    "mood": mood_id,
                    "mood_confidence": round(score, 3),
                })
            all_recs.extend(mood_recs)

        return {
            "caption": caption,
            "top_moods": [(m, round(s, 3)) for m, s in moods[:n_moods]],
            "recommendations": all_recs,
        }
 
 
def pretty_print(result: dict) -> None:
    print("\n" + "=" * 64)
    print(f"What I see: {result['caption']}")
    print("=" * 64)
 
    print("\nTop vibes detected:")
    for mood, score in result["top_moods"]:
        bar = "#" * int(score * 30)
        print(f"  {mood:20s} {bar} {score:.1%}")
 
    print("\nSong picks:")
    current_mood = None
    for r in result["recommendations"]:
        if r["mood"] != current_mood:
            current_mood = r["mood"]
            print(f"\n  --- {current_mood} ---")
        print(f"    {r['title']} - {r['artist']}")
        print(f"      {r['deezer_url']}")
    print()
 
 
def load_credentials() -> str:
    lastfm = os.environ.get("LASTFM_API_KEY")
    if not lastfm:
        sys.exit(
            "Missing env var: LASTFM_API_KEY\n"
            "Get a free key at: https://www.last.fm/api/account/create\n"
            "Then set it before running, e.g.:\n"
            "  export LASTFM_API_KEY=your_key_here   (macOS / Linux)\n"
            "  set LASTFM_API_KEY=your_key_here      (Windows cmd)"
        )
    return lastfm
 
 
def main() -> None:
    p = argparse.ArgumentParser(description="Suggest songs that match a photo's vibe.")
    p.add_argument("image", help="Path to image file")
    p.add_argument("--moods", type=int, default=2, help="How many top moods to use")
    p.add_argument("--songs", type=int, default=5, help="Songs per mood")
    args = p.parse_args()
 
    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")
 
    lastfm_key = load_credentials()
    tune = PhotoTune(LastFM(lastfm_key), Deezer())
    result = tune.recommend(args.image, n_moods=args.moods, songs_per_mood=args.songs)
    pretty_print(result)
 
 
if __name__ == "__main__":
    main()


