"""
PhotoTune Web UI (Deezer edition)

Drag-and-drop browser interface on top of phototune.py.

Setup:
    pip install gradio
    (plus everything in phototune.py: torch, transformers, pillow, requests)

    Set the env var as in phototune.py:
        LASTFM_API_KEY

Run:
    python phototune_ui.py
    -> opens at http://localhost:7860
"""

import gradio as gr

from phototune import Deezer, LastFM, PhotoTune, load_credentials


# Boot models + API clients ONCE at startup.
print("Booting PhotoTune...")
lastfm_key = load_credentials()
tune = PhotoTune(LastFM(lastfm_key), Deezer())
print("Ready.")


# =============================================================================
# Renderers
# =============================================================================
def render_caption(caption: str) -> str:
    return (
        '<div style="font-size:1.05em;font-style:italic;color:#444;'
        'padding:12px 16px;background:#f9f5ff;border-left:3px solid #a78bfa;'
        f'border-radius:4px;">"{caption}"</div>'
    )


def render_moods(top_moods: list[tuple[str, float]]) -> str:
    rows = []
    for mood, score in top_moods:
        pct = int(score * 100)
        label = mood.replace("_", " ").title()
        rows.append(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;
                        margin-bottom:4px;font-size:0.9em;">
                <span style="font-weight:500;">{label}</span>
                <span style="color:#888;">{pct}%</span>
            </div>
            <div style="height:6px;background:#eee;border-radius:3px;
                        overflow:hidden;">
                <div style="height:100%;width:{pct}%;
                            background:linear-gradient(90deg,#a78bfa,#ec4899);
                            border-radius:3px;"></div>
            </div>
        </div>
        """)
    return "".join(rows)


def render_songs(recommendations: list[dict]) -> str:
    if not recommendations:
        return ('<p style="color:#888;text-align:center;padding:20px;">'
                'No songs found - try another photo.</p>')

    cards = []
    for r in recommendations:
        art = r.get("album_art") or "https://via.placeholder.com/100?text=♪"
        mood_label = r["mood"].replace("_", " ").title()

        # Deezer almost always returns a 30s preview, so we render the audio
        # element unconditionally if the field is present.
        preview = ""
        if r.get("preview_url"):
            preview = (
                f'<audio controls src="{r["preview_url"]}" '
                'style="width:100%;height:32px;margin-top:8px;"></audio>'
            )

        cards.append(f"""
        <div style="border:1px solid #e5e5e5;border-radius:12px;padding:14px;
                    background:white;display:flex;gap:12px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <img src="{art}" alt="album art"
                 style="width:88px;height:88px;border-radius:8px;
                        object-fit:cover;flex-shrink:0;" />
            <div style="flex:1;min-width:0;display:flex;flex-direction:column;">
                <div style="font-weight:600;font-size:0.95em;
                            overflow:hidden;text-overflow:ellipsis;
                            white-space:nowrap;">{r['title']}</div>
                <div style="color:#666;font-size:0.85em;margin-bottom:4px;
                            overflow:hidden;text-overflow:ellipsis;
                            white-space:nowrap;">{r['artist']}</div>
                <div style="font-size:0.7em;color:#a78bfa;
                            text-transform:uppercase;letter-spacing:0.5px;
                            font-weight:600;margin-bottom:6px;">{mood_label}</div>
                <a href="{r['deezer_url']}" target="_blank"
                   style="color:#ef5466;text-decoration:none;font-size:0.85em;
                          font-weight:500;margin-top:auto;">
                    ▶ Open on Deezer →
                </a>
                {preview}
            </div>
        </div>
        """)

    return (
        '<div style="display:grid;'
        'grid-template-columns:repeat(auto-fill,minmax(320px,1fr));'
        f'gap:14px;">{"".join(cards)}</div>'
    )


# =============================================================================
# Main handler
# =============================================================================
def analyze(image_path, n_moods, n_songs):
    if not image_path:
        return ("", "", '<p style="color:#888;">Upload a photo first.</p>')
    try:
        result = tune.recommend(
            image_path,
            n_moods=int(n_moods),
            songs_per_mood=int(n_songs),
        )
        return (
            render_caption(result["caption"]),
            render_moods(result["top_moods"]),
            render_songs(result["recommendations"]),
        )
    except Exception as e:
        err = (f'<div style="color:#c33;padding:12px;background:#fee;'
               f'border-radius:8px;">Something went wrong: {e}</div>')
        return ("", "", err)


# =============================================================================
# UI layout
# =============================================================================
INTRO = """
# 🎵 PhotoTune

Upload a photo and get song suggestions that match its vibe.
Built for picking music for Instagram posts, stories, and reels.
"""

with gr.Blocks(title="PhotoTune") as app:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="Drop a photo here",
                height=380,
            )
            with gr.Row():
                n_moods = gr.Slider(
                    1, 4, value=2, step=1,
                    label="How many vibes to detect",
                )
                n_songs = gr.Slider(
                    2, 8, value=4, step=1,
                    label="Songs per vibe",
                )
            submit = gr.Button("Find songs ✨", variant="primary", size="lg")
            gr.Markdown(
                "<small style='color:#888;'>Tip: click again for fresh picks - "
                "results are randomised from the candidate pool.</small>"
            )

        with gr.Column(scale=2):
            caption_output = gr.HTML()
            gr.Markdown("### 🎨 Top vibes")
            moods_output = gr.HTML()
            gr.Markdown("### 🎶 Song picks")
            songs_output = gr.HTML()

    submit.click(
        analyze,
        inputs=[image_input, n_moods, n_songs],
        outputs=[caption_output, moods_output, songs_output],
    )


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink"))
