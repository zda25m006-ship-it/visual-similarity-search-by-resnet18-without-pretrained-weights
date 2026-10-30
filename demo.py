import argparse
import base64
import io
import os
import time
import numpy as np
from PIL import Image
import torch

from data import build_cifar100_datasets, get_transforms
from model import ResNet18Embedder
from retrieval import BruteForceRetriever, extract_embeddings_to_numpy


def _pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VisualSearchEngineApp:
    """Inference wrapper loading the trained ResNet-18 model and test gallery index."""

    def __init__(self, checkpoint_path="./artifacts/best_resnet18_triplet.pt", data_root="./data", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datasets, self.class_names = build_cifar100_datasets(data_root=data_root, seed=42)
        self.eval_transform = get_transforms(train=False)

        embedding_dim = 128
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            embedding_dim = int(ckpt.get("embedding_dim", 128))
            self.model = ResNet18Embedder(embedding_dim=embedding_dim, cifar_stem=True).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model = ResNet18Embedder(embedding_dim=embedding_dim, cifar_stem=True).to(self.device)

        self.model.eval()

        emb_file = os.path.join(os.path.dirname(checkpoint_path) or ".", "test_embeddings.npy")
        lbl_file = os.path.join(os.path.dirname(checkpoint_path) or ".", "test_labels.npy")
        if os.path.exists(emb_file) and os.path.exists(lbl_file):
            self.gallery_embeddings = np.load(emb_file)
            self.gallery_labels = np.load(lbl_file)
        else:
            from torch.utils.data import DataLoader
            loader = DataLoader(self.datasets["test"], batch_size=256, shuffle=False)
            self.gallery_embeddings, self.gallery_labels, _ = extract_embeddings_to_numpy(self.model, loader, self.device)

        self.retriever = BruteForceRetriever(self.gallery_embeddings, self.gallery_labels, self.class_names)

    @torch.no_grad()
    def embed_pil_image(self, pil_img):
        img_rgb = pil_img.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        tensor_in = self.eval_transform(img_rgb).unsqueeze(0).to(self.device)
        return self.model(tensor_in).cpu().numpy()[0]

    def search_top_k(self, pil_img, top_k=5, exclude_index=None):
        t0 = time.perf_counter()
        q_emb = self.embed_pil_image(pil_img)
        ts = time.perf_counter()

        exclude_arr = np.array([exclude_index]) if exclude_index is not None else None
        res = self.retriever.search_l2(q_emb, top_k=top_k, exclude_self_indices=exclude_arr)
        search_ms = (time.perf_counter() - ts) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        items = []
        for r_i in range(top_k):
            g_idx = int(res["indices"][0, r_i])
            dist = float(res["distances"][0, r_i])
            cos_s = float(res["cosine_similarities"][0, r_i])
            cls_id = int(res["labels"][0, r_i])
            cls_name = self.class_names[cls_id].replace("_", " ").title()

            raw_gal = self.datasets["test"].get_raw_image(g_idx)
            gal_pil = Image.fromarray(raw_gal).resize((280, 280), Image.Resampling.BICUBIC)
            items.append({
                "rank": r_i + 1,
                "class_name": cls_name,
                "l2": dist,
                "cosine": cos_s,
                "image": gal_pil,
            })

        return {
            "items": items,
            "search_ms": search_ms,
            "total_ms": total_ms,
            "excluded": 1 if exclude_index is not None else 0,
        }


STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #efefed;
    color: #20242a;
}
.block-container {
    max-width: 1380px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
}
.nn-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0 1.1rem 0;
    border-bottom: 1px solid #ddddd8;
    margin-bottom: 1.75rem;
}
.nn-brand {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: #20242a;
}
.nn-brand span { color: #285ed7; }
.nn-mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #6c727d;
}
.nn-hero {
    background: linear-gradient(180deg, #e5e5e3 0%, #e7e7e5 100%);
    border-radius: 22px;
    min-height: 390px;
    padding: 2.2rem 2.5rem;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    margin-bottom: 2.75rem;
}
.nn-hero-mid {
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    min-height: 270px;
}
.nn-hero-bg-text {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: clamp(4.5rem, 11vw, 9.2rem);
    font-weight: 700;
    line-height: 0.86;
    letter-spacing: -0.045em;
    color: #f9f9f8;
    user-select: none;
    pointer-events: none;
}
.nn-orb {
    width: 210px;
    height: 210px;
    border-radius: 9999px;
    margin: 0 auto;
    z-index: 2;
    background:
        radial-gradient(circle at 36% 34%, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.65) 26%, rgba(255,255,255,0) 62%),
        conic-gradient(from 205deg at 50% 50%, #d4dff2 0deg, #edd0e5 65deg, #d1dbf2 135deg, #eaf3fa 205deg, #f3dfcf 285deg, #d4dff2 360deg);
    border: 1.5px solid rgba(255,255,255,0.92);
    box-shadow: 0 20px 44px rgba(40,52,74,0.11), inset 0 0 22px rgba(255,255,255,0.85);
    position: relative;
}
.nn-orb::after {
    content: "";
    position: absolute;
    inset: 7px;
    border-radius: 9999px;
    border: 1px solid rgba(255,255,255,0.78);
}
.nn-side-note {
    z-index: 3;
    color: #285ed7;
    font-size: 0.82rem;
    line-height: 1.45;
}
.nn-pill-cta {
    margin: 0 auto;
    z-index: 3;
    background: #ffffff;
    color: #285ed7;
    border: 1px solid #dfdfe2;
    border-radius: 9999px;
    padding: 0.5rem 1.25rem;
    font-size: 0.78rem;
    font-weight: 500;
    box-shadow: 0 4px 14px rgba(32,36,42,0.045);
}
.nn-ws-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}
.nn-ws-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #20242a;
    margin: 0;
}
.nn-ready-pill {
    background: #ffffff;
    border: 1px solid #e1e1dd;
    border-radius: 9999px;
    padding: 0.35rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    color: #454b54;
}
.nn-ready-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2e9b63;
    margin-right: 6px;
}
.nn-info-card {
    background: #ffffff;
    border-radius: 18px;
    border: 1px solid #e4e4e0;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.25rem;
}
.nn-info-card h4 {
    margin: 0 0 0.35rem 0;
    font-size: 0.98rem;
    font-weight: 700;
    color: #20242a;
}
.nn-info-card p {
    margin: 0;
    font-size: 0.87rem;
    color: #68707c;
}
.nn-img-wrap img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: 18px;
    display: block;
}
.nn-meta-card {
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #e5e5e1;
    padding: 1rem 1.15rem;
    margin-top: 0.75rem;
}
.nn-meta-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    color: #6c727d;
    margin-bottom: 0.35rem;
}
.nn-meta-cls {
    font-size: 1.05rem;
    font-weight: 700;
    color: #20242a;
    margin-bottom: 0.45rem;
}
.nn-meta-l2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6c727d;
}
</style>
"""


def run_streamlit_demo(checkpoint_path="./artifacts/best_resnet18_triplet.pt", data_root="./data"):
    """Launch the Streamlit web interface."""
    import streamlit as st

    st.set_page_config(page_title="near/near · CIFAR-100", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)

    @st.cache_resource
    def _get_engine():
        return VisualSearchEngineApp(checkpoint_path=checkpoint_path, data_root=data_root)

    engine = _get_engine()

    st.markdown(
        """
        <div class="nn-topbar">
            <div class="nn-brand">near/<span>near</span></div>
            <div class="nn-mono">A SMALL STUDY IN VISUAL SIMILARITY &middot; CIFAR-100</div>
            <div class="nn-mono">METRIC LEARNING STUDIO &#8599;</div>
        </div>
        <div class="nn-hero">
            <div class="nn-hero-mid">
                <div class="nn-side-note">[ From pixels<br/>to neighborhoods ]</div>
                <div class="nn-hero-bg-text"><div>LOOK</div><div>&amp; FIND</div></div>
                <div class="nn-orb"></div>
                <div class="nn-side-note">Find images that share a visual neighborhood.</div>
            </div>
            <div class="nn-pill-cta">Upload a query to explore the gallery &#8599;</div>
        </div>
        <div class="nn-mono" style="margin-bottom: 0.4rem;">01 / SEARCH WORKSPACE</div>
        <div class="nn-ws-header">
            <h2 class="nn-ws-title">Find your nearest neighbors.</h2>
            <div class="nn-ready-pill"><span class="nn-ready-dot"></span>MODEL + INDEX READY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="nn-info-card"><h4>Your query image</h4><p>Upload a picture to search the CIFAR-100 gallery.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="nn-info-card"><h4>How the search works</h4><p>ResNet-18 &rarr; normalized 128-D embedding &rarr; exact NumPy L2 search. Five neighbors appear with class labels, distances, and cosine similarity.</p></div>',
            unsafe_allow_html=True,
        )

    left_col, right_col = st.columns([2, 1])
    with left_col:
        mode = st.radio("Query source", ["Upload an image", "Sample from CIFAR-100 test gallery"], index=1, horizontal=True)
        exclude_self = st.checkbox("Exclude exact self match", value=False)
        if mode == "Upload an image":
            up = st.file_uploader("Upload query image", type=["png", "jpg", "jpeg"])
            if up is not None:
                query_pil = Image.open(up).convert("RGB")
                exclude_idx = None
            else:
                query_pil = Image.fromarray(engine.datasets["test"].get_raw_image(0))
                exclude_idx = 0 if exclude_self else None
        else:
            q_idx = st.slider("Select CIFAR-100 test image index", 0, len(engine.datasets["test"]) - 1, 0)
            query_pil = Image.fromarray(engine.datasets["test"].get_raw_image(q_idx))
            exclude_idx = q_idx if exclude_self else None

    with right_col:
        st.image(query_pil.resize((160, 160), Image.Resampling.BICUBIC), caption="Active Query", width=140)

    res = engine.search_top_k(query_pil, top_k=5, exclude_index=exclude_idx)

    st.markdown(
        f"""
        <div style="margin-top: 2rem; margin-bottom: 1.25rem;">
            <h2 class="nn-ws-title" style="margin-bottom: 0.45rem;">Five images, one visual space.</h2>
            <div style="font-size: 0.85rem; color: #6c727d;">
                Sorted by ascending L2 distance &middot; search {res['search_ms']:.1f} ms &middot; total {res['total_ms']:.1f} ms &middot; exact self matches excluded: {res['excluded']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    for idx, item in enumerate(res["items"]):
        b64 = _pil_to_base64(item["image"])
        with cols[idx]:
            st.markdown(
                f"""
                <div class="nn-img-wrap"><img src="data:image/png;base64,{b64}" alt="{item['class_name']}" /></div>
                <div class="nn-meta-card">
                    <div class="nn-meta-rank">#{item['rank']:02d} / COSINE {item['cosine']:.3f}</div>
                    <div class="nn-meta-cls">{item['class_name']}</div>
                    <div class="nn-meta-l2">L2 {item['l2']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def launch_gradio_demo(checkpoint_path="./artifacts/best_resnet18_triplet.pt", data_root="./data", share=True, server_port=7860):
    """Launch the Gradio web interface."""
    import gradio as gr

    engine = VisualSearchEngineApp(checkpoint_path=checkpoint_path, data_root=data_root)

    ex_dir = "./artifacts/demo_examples"
    os.makedirs(ex_dir, exist_ok=True)
    examples = []
    for idx in [0, 12, 145, 512, 1024]:
        raw = engine.datasets["test"].get_raw_image(idx)
        cname = engine.class_names[int(engine.datasets["test"].labels[idx])]
        p = os.path.join(ex_dir, f"test_{idx}_{cname}.png")
        Image.fromarray(raw).resize((160, 160), Image.Resampling.BICUBIC).save(p)
        examples.append(p)

    def _predict(img, exclude_self):
        if img is None:
            return "<p>Select a query image.</p>"
        res = engine.search_top_k(img, top_k=5, exclude_index=0 if exclude_self else None)
        cards = []
        for item in res["items"]:
            b64 = _pil_to_base64(item["image"])
            cards.append(
                f"""
                <div style="flex:1; min-width:170px;">
                    <img src="data:image/png;base64,{b64}" style="width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:18px;" />
                    <div style="background:#ffffff; border-radius:16px; border:1px solid #e5e5e1; padding:14px 16px; margin-top:10px;">
                        <div style="font-family:monospace; font-size:11px; color:#6c727d;">#{item['rank']:02d} / COSINE {item['cosine']:.3f}</div>
                        <div style="font-size:17px; font-weight:700; color:#20242a; margin:4px 0 6px 0;">{item['class_name']}</div>
                        <div style="font-family:monospace; font-size:11.5px; color:#6c727d;">L2 {item['l2']:.4f}</div>
                    </div>
                </div>
                """
            )
        return f"""
        <div style="background:#efefed; padding:20px; border-radius:20px; font-family:Inter, sans-serif;">
            <h2 style="font-size:26px; font-weight:700; color:#20242a; margin:0 0 6px 0;">Five images, one visual space.</h2>
            <p style="font-size:13px; color:#6c727d; margin:0 0 18px 0;">
                Sorted by ascending L2 distance &middot; search {res['search_ms']:.1f} ms &middot; total {res['total_ms']:.1f} ms &middot; exact self matches excluded: {res['excluded']}
            </p>
            <div style="display:flex; gap:16px; flex-wrap:wrap;">{''.join(cards)}</div>
        </div>
        """

    with gr.Blocks(title="near/near · CIFAR-100") as demo:
        gr.HTML(
            STYLES
            + """
            <div class="nn-topbar">
                <div class="nn-brand">near/<span>near</span></div>
                <div class="nn-mono">A SMALL STUDY IN VISUAL SIMILARITY &middot; CIFAR-100</div>
                <div class="nn-mono">METRIC LEARNING STUDIO &#8599;</div>
            </div>
            <div class="nn-hero">
                <div class="nn-hero-mid">
                    <div class="nn-side-note">[ From pixels<br/>to neighborhoods ]</div>
                    <div class="nn-hero-bg-text"><div>LOOK</div><div>&amp; FIND</div></div>
                    <div class="nn-orb"></div>
                    <div class="nn-side-note">Find images that share a visual neighborhood.</div>
                </div>
                <div class="nn-pill-cta">Upload a query to explore the gallery &#8599;</div>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                q_in = gr.Image(type="pil", label="Your query image")
                ex_chk = gr.Checkbox(label="Exclude exact self match", value=False)
                btn = gr.Button("Find your nearest neighbors", variant="primary")
                gr.Examples(examples=examples, inputs=q_in, label="CIFAR-100 Test Samples")
            with gr.Column(scale=3):
                out_html = gr.HTML()

        btn.click(fn=_predict, inputs=[q_in, ex_chk], outputs=out_html)
        q_in.change(fn=_predict, inputs=[q_in, ex_chk], outputs=out_html)

    demo.launch(share=share, server_port=server_port, inline=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", type=str, default="streamlit", choices=["streamlit", "gradio"])
    parser.add_argument("--checkpoint", type=str, default="./artifacts/best_resnet18_triplet.pt")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--share", action="store_true", default=True)
    parser.add_argument("--port", type=int, default=7860)
    args, _ = parser.parse_known_args()

    if args.ui == "gradio":
        launch_gradio_demo(checkpoint_path=args.checkpoint, data_root=args.data_root, share=args.share, server_port=args.port)
    else:
        run_streamlit_demo(checkpoint_path=args.checkpoint, data_root=args.data_root)
