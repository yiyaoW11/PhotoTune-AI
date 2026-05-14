# 🎵 PhotoTune AI

> **Drop a photo, get music that matches the vibe.**
> A multimodal AI app that uses computer vision to suggest songs for Instagram posts, stories, and reels.

---

## 🌟 What it does

Picking the right song for a photo is harder than it sounds. PhotoTune AI solves this problem by *seeing* your photo and matching it to music with the same mood. Each suggestion comes with a 30-second preview and a clickable Deezer link.

---

## 🧠 How it works

PhotoTune chains four AI / API components into one pipeline:

```
        Photo
          │
          ▼
   ┌──────────────┐     ┌──────────────┐
   │     BLIP     │     │     CLIP     │
   │  (caption)   │     │  (mood score)│
   └──────────────┘     └──────────────┘
          │                     │
          │                     ▼
          │            ┌──────────────────┐
          │            │     Last.fm      │
          │            │ (top tracks per  │
          │            │   mood tag)      │
          │            └──────────────────┘
          │                     │
          │                     ▼
          │            ┌──────────────────┐
          │            │      Deezer      │
          │            │ (playable links, │
          │            │  previews, art)  │
          │            └──────────────────┘
          │                     │
          ▼                     ▼
       Caption          Song recommendations
```

| Step | Model / API | What it does |
|------|-------------|-------------|
| 1 | **BLIP** (Salesforce) | Generates a one-sentence description of the photo |
| 2 | **CLIP** (OpenAI) | Scores the photo against 12 mood descriptions in a shared image-text embedding space |
| 3 | **Last.fm** | Pulls popular songs tagged with the top-matching moods (crowd-sourced from millions of users) |
| 4 | **Deezer** | Enriches each track with playable URLs, 30-second previews, and album art |

Both BLIP and CLIP are pre-trained transformer-based neural networks — no training data needed. CLIP's zero-shot classification does the heavy lifting: by writing 12 descriptive sentences like *"a dreamy romantic photo with a soft aesthetic,"* we can categorize any photo into a mood without ever training a custom model.

---
# Evaluation
 
A quantitative evaluation of PhotoTune's mood classification accuracy on a test set of Unsplash photos. 
 
---
 
## TL;DR
 
| Metric | Score |
|--------|------:|
| **Top-1 accuracy** | **70.8%** |
| **Top-3 accuracy** | **96.9%** |
| Random baseline (12 classes) | 8.3% / 25.0% |
 
PhotoTune is **8.5× better than random** on top-1. Top-3 accuracy of 96.9% means the correct mood is almost always among the top predictions — which is why PhotoTune blends songs from the top 2 moods in production rather than committing to one. 

## Methodology
### Dataset
 
A balanced test set of **96 photos** was collected from Unsplash — 8 photos per mood across all 12 categories. Photos were organized into folders named after their mood (ImageFolder convention), with the folder name serving as the label. Each photo was hand-selected as a clear example of its mood.


## 🛠️ Tech stack

- **Python 3.10+**
- **PyTorch** + **Hugging Face Transformers** (BLIP, CLIP)
- **Gradio** — drag-and-drop web UI
- **Pillow** + `pillow-avif-plugin` — image handling
- **Requests** — Last.fm + Deezer API calls
- **python-dotenv** — environment variable loading

---

## 🚀 Setup

### 1. Clone the repo

```bash
git clone https://github.com/yiyaoW11/PhotoTune-AI.git
cd PhotoTune-AI
```

### 2. Install dependencies

```bash
pip install torch transformers pillow requests python-dotenv gradio pillow-avif-plugin
```

### 3. Get a free Last.fm API key

Sign up at https://www.last.fm/api/account/create (takes 30 seconds, no callback URL needed).

### 4. Create a `.env` file

In the project folder, create a file called `.env` with your key:

```
LASTFM_API_KEY=your_key_here
```

### 5. Run it

For the web UI (recommended):

```bash
python phototune_ui.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

For a command-line one-shot:

```bash
python phototune.py path/to/your/photo.jpg
```

> ⏱️ The first run downloads ~1GB of model weights from Hugging Face. Subsequent runs load from cache and start in seconds.

---

## 📁 Project structure

```
PhotoTune-AI/
├── phototune.py        # Core pipeline (vision models + API clients)
├── phototune_ui.py     # Gradio web interface
├── moods.py            # Mood vocabulary and Last.fm tag mappings
├── .env                # Your API key (gitignored)
├── .gitignore
└── README.md
```

The code is structured with three main classes:

- `LastFM` — handles top-tracks-per-tag lookups
- `Deezer` — handles song search and metadata enrichment
- `PhotoTune` — orchestrates the vision models and API clients

Each class encapsulates one concern, making it straightforward to swap APIs (e.g., replacing Deezer with Apple Music) without changing the rest of the pipeline.

---

## 🎨 Mood vocabulary

The system supports 12 moods out of the box, each mapped to a CLIP-friendly descriptive sentence and 2–3 Last.fm tags:

| Mood | Example Last.fm tags |
|------|---------------------|
| happy_upbeat | happy, feel good, upbeat |
| chill_relaxed | chillout, chill, relaxing |
| romantic_dreamy | romantic, dream pop, love songs |
| moody_dark | dark, moody, atmospheric |
| nostalgic | nostalgia, 80s, oldies |
| energetic_party | party, dance, club |
| ethereal | ethereal, dream pop, ambient |
| cozy_intimate | acoustic, singer-songwriter, mellow |
| adventurous | indie folk, epic, uplifting |
| edgy_cool | indie rock, alternative, garage rock |
| melancholic | melancholy, sad, bittersweet |
| fun_playful | fun, indie pop, feel good |

New moods can be added by editing the `moods.py` file

---

## 🧩 Design decisions

**Why Deezer instead of Spotify?**
Spotify made their Web API a Premium-only feature in February 2026. Deezer's search API is fully public, no signup required — and it consistently returns 30-second previews where Spotify often doesn't.

**Why both BLIP and CLIP?**
They complement each other. CLIP does the actual mood matching (its strength). BLIP provides a human-readable description for transparency — so the user understands *why* certain songs were suggested.

**Why Last.fm in the middle?**
Searching Spotify or Deezer for "moody" returns songs with "moody" *in the title*. Last.fm tags come from millions of users actually listening to and labeling music, making them far better for mood-based discovery.

---

## 🔮 Future improvements

- **Fine-tune CLIP** on Instagram-specific aesthetics (cottagecore, y2k, clean girl, dark academia) for sharper niche detection
- **Personalisation layer** — track which suggestions get clicked and bias future picks accordingly (would make this a real recommender system with implicit feedback)
- **Deploy publicly** on Hugging Face Spaces so anyone can try it without local setup
- **Multiple-photo carousels** — analyze all photos in a post and find one song that fits the whole set
- **Lyric-aware matching** — embed BLIP's caption and song lyrics to find tracks whose *content* (not just mood) matches the photo

---

## 📜 Credits

- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) by Salesforce Research
- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) by OpenAI
- [Last.fm API](https://www.last.fm/api) for tag-based song data
- [Deezer API](https://developers.deezer.com/api) for playable music metadata

---

## 📄 License

MIT — feel free to fork, modify, and build on this.
