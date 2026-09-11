import os
import sys
import glob
import tempfile
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import plotly.graph_objects as go
from PIL import Image

# ------------------------------------------------------------------------------
# 1. Setup paths to include OpenGait-2.0 modules
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "OpenGait-2.0")):
    OPENGAIT_DIR = os.path.join(CURRENT_DIR, "OpenGait-2.0")
else:
    OPENGAIT_DIR = CURRENT_DIR

OPENGAIT_PKG_DIR = os.path.join(OPENGAIT_DIR, "opengait")

if OPENGAIT_DIR not in sys.path:
    sys.path.insert(0, OPENGAIT_DIR)
if OPENGAIT_PKG_DIR not in sys.path:
    sys.path.insert(0, OPENGAIT_PKG_DIR)

import yaml

def load_yaml_config(cfg_path):
    """Loads OpenGait YAML configuration and merges default config parameters."""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        src_cfgs = yaml.safe_load(f)
    default_cfg_path = os.path.join(OPENGAIT_DIR, "configs", "default.yaml")
    if os.path.exists(default_cfg_path):
        with open(default_cfg_path, 'r', encoding='utf-8') as f:
            dst_cfgs = yaml.safe_load(f)
        def merge_dicts(src, dst):
            for k, v in src.items():
                if k not in dst or not isinstance(v, dict):
                    dst[k] = v
                else:
                    if isinstance(src[k], dict) and isinstance(dst[k], dict):
                        merge_dicts(src[k], dst[k])
                    else:
                        dst[k] = v
        merge_dicts(dst_cfgs, src_cfgs)
        return src_cfgs
    return src_cfgs

def _load_baseline_class():
    """Load the Baseline model class without triggering the full models package import.

    opengait.modeling.models.__init__ imports every model (gaitedge, gln, smplgait, ...),
    some of which pull in optional dependencies. Importing the baseline module directly
    via importlib keeps inference robust.
    """
    import importlib.util
    baseline_path = os.path.join(OPENGAIT_DIR, "opengait", "modeling", "models", "baseline.py")
    if not os.path.exists(baseline_path):
        raise ImportError(f"Baseline model module not found at: {baseline_path}")
    spec = importlib.util.spec_from_file_location(
        "opengait.modeling.models.baseline", baseline_path)
    baseline_mod = importlib.util.module_from_spec(spec)
    sys.modules["opengait.modeling.models.baseline"] = baseline_mod
    spec.loader.exec_module(baseline_mod)
    return baseline_mod.Baseline

try:
    Baseline = _load_baseline_class()
except Exception as e:
    try:
        from opengait.modeling.models.baseline import Baseline  # type: ignore
    except Exception:
        raise ImportError(f"Could not load Baseline model module: {e}")

# Default checkpoint and config paths
_default_ckpt_rel = os.path.join("pretrained_casiab_gaitbase", "CASIA-B", "Baseline", "GaitBase_DA", "checkpoints", "GaitBase_DA-60000.pt")
DEFAULT_CKPT_PATH = os.path.join(CURRENT_DIR, "weights", _default_ckpt_rel)
if not os.path.exists(DEFAULT_CKPT_PATH):
    DEFAULT_CKPT_PATH = os.path.join(CURRENT_DIR, _default_ckpt_rel)
DEFAULT_CFG_PATH = os.path.join(
    OPENGAIT_DIR, "configs", "gaitbase", "gaitbase_da_casiab.yaml"
)

# ------------------------------------------------------------------------------
# 2. Standalone Model Wrapper for Inference (No DDP required)
# ------------------------------------------------------------------------------
class OpenGaitInferencer:
    def __init__(self, cfg_path=DEFAULT_CFG_PATH, ckpt_path=DEFAULT_CKPT_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.model = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f"Config file not found at: {self.cfg_path}")
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found at: {self.ckpt_path}")

        print(f"Loading OpenGait config from {self.cfg_path}...")
        cfgs = load_yaml_config(self.cfg_path)
        model_cfg = cfgs['model_cfg']

        # Instantiate Baseline module directly without DDP initialization
        print("Instantiating OpenGait Baseline network architecture...")
        model_module = Baseline.__new__(Baseline)
        nn.Module.__init__(model_module)
        model_module.build_network(model_cfg)

        # Load weights
        print(f"Loading checkpoint weights from {self.ckpt_path}...")
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model_module.load_state_dict(state_dict, strict=True)

        model_module.to(self.device)
        model_module.eval()
        self.model = model_module
        print("OpenGait model loaded successfully!")

    @torch.no_grad()
    def extract_feature(self, sil_seq):
        """
        Args:
            sil_seq: numpy array of shape (T, 64, 64) with uint8 silhouette masks [0, 255]
        Returns:
            feature tensor of shape (1, 256, 16)
        """
        # Apply BaseSilCuttingTransform: cut 10 pixels from left and right side (64x64 -> 64x44)
        cutting = int(sil_seq.shape[-1] // 64) * 10  # 10
        sil_cut = sil_seq[..., cutting:-cutting]     # (T, 64, 44)
        sil_norm = sil_cut.astype(np.float32) / 255.0

        # Convert to tensor [B, 1, T, H, W]
        ipts = torch.from_numpy(sil_norm).unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, T, 64, 44]

        # Model forward
        # Backbone (SetBlockWrapper) -> [1, 512, T, 16, 11]
        outs = self.model.Backbone(ipts)
        # Temporal Pooling (TP) -> [1, 512, 16, 11]
        outs = self.model.TP(outs, None, options={"dim": 2})[0]
        # Horizontal Pooling Pyramid (HPP) -> [1, 512, 16]
        feat = self.model.HPP(outs)
        # Separate FCs -> [1, 256, 16]
        embed = self.model.FCs(feat)
        return embed

# Initialize Inferencer singleton lazily
INFERENCER = None

def get_inferencer():
    global INFERENCER
    if INFERENCER is None:
        INFERENCER = OpenGaitInferencer()
    return INFERENCER

# ------------------------------------------------------------------------------
# 3. Silhouette Extraction & Preprocessing Utilities
# ------------------------------------------------------------------------------
def process_video_to_silhouettes(video_path, max_frames=60):
    """
    Extracts a silhouette frame sequence from an input video and resizes each frame to (64, 64)
    while preserving the person's aspect ratio (OpenGait is trained on 64x44 aligned silhouettes).

    Raises ValueError if no valid silhouette could be extracted (instead of returning dummy frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frames = []
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        # MOG2 outputs 0 / 127 (shadow) / 255. Keep the real foreground (255).
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Find largest contour (person silhouette)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300:  # filter noise
                x, y, w, h = cv2.boundingRect(c)
                # Crop around the person silhouette, then resize preserving aspect ratio
                # by padding the shorter side (avoid stretching the gait proportions).
                crop = thresh[y:y + h, x:x + w]
                side = max(w, h)
                pad_x = (side - w) // 2
                pad_y = (side - h) // 2
                padded = cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x,
                                            cv2.BORDER_CONSTANT, value=0)
                resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_NEAREST)
                frames.append(resized)

    cap.release()
    if not frames:
        raise ValueError(
            "No person silhouette could be extracted from the video. "
            "Try a clip with a clear, moving subject on a static background, "
            "or upload pre-segmented silhouette images instead."
        )
    return np.array(frames, dtype=np.uint8)


def _resolve_file_paths(file_list):
    """Extract server file paths from Gradio's FileData objects or plain strings."""
    paths = []
    for f in file_list:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "path"):
            paths.append(f.path)
        elif isinstance(f, dict) and "path" in f:
            paths.append(f["path"])
        else:
            # Fallback: some Gradio versions expose the path differently.
            paths.append(str(f))
    return paths

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg")


def load_sequence(uploaded):
    """Normalize a Gradio upload into a silhouette sequence by extension."""
    if uploaded is None:
        raise ValueError("No file was uploaded.")
    if not isinstance(uploaded, list):
        uploaded = [uploaded]
    paths = _resolve_file_paths(uploaded)
    if not paths:
        raise ValueError("No file was uploaded.")
    video_paths = [p for p in paths if p.lower().endswith(_VIDEO_EXTS)]
    image_paths = [p for p in paths if not p.lower().endswith(_VIDEO_EXTS)]
    if image_paths:
        return process_image_folder_or_files(image_paths)
    if video_paths:
        return process_video_to_silhouettes(video_paths[0])
    try:
        return process_video_to_silhouettes(paths[0])
    except Exception:
        return process_image_folder_or_files(paths)

def process_image_folder_or_files(image_files, max_frames=60):
    """
    Reads multiple uploaded silhouette images and formats them into a (T, 64, 64) numpy sequence.
    Silhouette images are scaled preserving aspect ratio (padded, not stretched).

    Raises ValueError if no readable silhouette image is found.
    """
    frames = []
    for file_path in _resolve_file_paths(image_files)[:max_frames]:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        side = max(w, h)
        pad_x = (side - w) // 2
        pad_y = (side - h) // 2
        padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                    cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_NEAREST)
        frames.append(resized)
    if not frames:
        raise ValueError(
            "No readable silhouette image was found among the uploaded files."
        )
    return np.array(frames, dtype=np.uint8)

def create_preview_gif(sil_seq):
    """Generates a preview GIF from silhouette sequence for UI visualization."""
    temp_dir = tempfile.gettempdir()
    gif_path = os.path.join(temp_dir, f"sil_preview_{np.random.randint(10000, 99999)}.gif")

    images = [Image.fromarray(frame) for frame in sil_seq]
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
        return gif_path
    return None

# ------------------------------------------------------------------------------
# 4. Metric Computation & Visualization
# ------------------------------------------------------------------------------
def compute_metrics(embed1, embed2):
    """
    Computes part-wise Euclidean distances and overall similarity.
    embed1, embed2 shape: (1, 256, 16)
    """
    # 16 horizontal bins (parts)
    num_bins = embed1.size(2)
    # Part-wise Euclidean distances across 256-dim feature vectors
    diff = torch.sqrt(torch.sum((embed1 - embed2) ** 2, dim=1) + 1e-9).squeeze(0) # shape (16,)
    part_distances = diff.cpu().numpy()
    mean_dist = float(part_distances.mean())

    # Cosine Similarity (scale-independent; the more reliable metric for GaitBase features)
    e1_norm = F.normalize(embed1, p=2, dim=1)
    e2_norm = F.normalize(embed2, p=2, dim=1)
    cos_sim = float((e1_norm * e2_norm).sum(dim=1).mean().cpu().item())

    # Confidence score derived from cosine similarity, mapped to 0-100%.
    # cos_sim ranges roughly [-1, 1]; same-person pairs are typically well above 0.
    match_score = max(0.0, min(100.0, (cos_sim + 1.0) / 2.0 * 100.0))

    return mean_dist, cos_sim, match_score, part_distances

def generate_part_plot(part_distances):
    """Creates a 16-part horizontal body segment distance bar chart using Plotly."""
    body_parts = [
        f"Part {i+1} ({['Head', 'Shoulders', 'Upper Chest', 'Lower Chest', 'Waist', 'Hips', 'Upper Thighs', 'Mid Thighs', 'Knees', 'Lower Knees', 'Upper Shins', 'Mid Shins', 'Ankles', 'Lower Ankles', 'Feet', 'Base'][i]})"
        for i in range(16)
    ]
    fig = go.Figure(go.Bar(
        x=part_distances,
        y=body_parts,
        orientation='h',
        marker=dict(
            color=part_distances,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Euclidean Dist")
        )
    ))
    fig.update_layout(
        title="16-Part Horizontal Body Feature Distance Breakdown",
        xaxis_title="Euclidean Distance (Lower = More Similar)",
        yaxis_title="Body Segment",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark"
    )
    return fig

# ------------------------------------------------------------------------------
# 5. Core Frontend Handlers
# ------------------------------------------------------------------------------
GALLERY_DB = {} # Memory gallery dictionary: {subject_name: feature_tensor}

def verify_gait_pair(probe_file, gallery_file, threshold):
    """Handler for 1-vs-1 Verification tab."""
    try:
        inferencer = get_inferencer()
    except Exception as e:
        return None, None, f"<h3 style='color:red;'>Model Load Error: {str(e)}</h3>", None

    if probe_file is None or gallery_file is None:
        return None, None, "<h3 style='color:orange;'>Please upload both Probe and Gallery video/image sequences!</h3>", None

    # Load probe sequence
        # Load probe sequence
    seq1 = load_sequence(probe_file)

    # Load gallery sequence
    seq2 = load_sequence(gallery_file)

    # Extract features
    feat1 = inferencer.extract_feature(seq1)
    feat2 = inferencer.extract_feature(seq2)

    # Compute metrics
    mean_dist, cos_sim, match_score, part_dists = compute_metrics(feat1, feat2)

    is_match = cos_sim >= threshold

    # Decision Badge
    if is_match:
        badge_html = f"""
        <div style='background-color: #1e3a1e; border: 2px solid #4caf50; border-radius: 10px; padding: 20px; text-align: center;'>
            <h2 style='color: #4caf50; margin: 0;'>VERDICT: MATCH (SAME PERSON)</h2>
            <p style='font-size: 18px; color: #e0e0e0; margin-top: 10px;'>
                Cosine Similarity: <b>{cos_sim:.4f}</b> (Threshold: {threshold:.2f})<br>
                Euclidean Distance: <b>{mean_dist:.4f}</b> | Confidence: <b>{match_score:.1f}%</b>
            </p>
        </div>
        """
    else:
        badge_html = f"""
        <div style='background-color: #3a1e1e; border: 2px solid #f44336; border-radius: 10px; padding: 20px; text-align: center;'>
            <h2 style='color: #f44336; margin: 0;'>VERDICT: MISMATCH (DIFFERENT PERSON)</h2>
            <p style='font-size: 18px; color: #e0e0e0; margin-top: 10px;'>
                Cosine Similarity: <b>{cos_sim:.4f}</b> (Threshold: {threshold:.2f})<br>
                Euclidean Distance: <b>{mean_dist:.4f}</b> | Confidence: <b>{match_score:.1f}%</b>
            </p>
        </div>
        """

    gif1 = create_preview_gif(seq1)
    gif2 = create_preview_gif(seq2)
    part_fig = generate_part_plot(part_dists)

    return gif1, gif2, badge_html, part_fig

def register_identity(subject_id, video_or_files):
    """Registers a subject into memory gallery."""
    if not subject_id.strip():
        return "Please enter a valid Subject ID!"
    if video_or_files is None:
        return "Please upload a video or sequence of images!"

    try:
        inferencer = get_inferencer()
        seq = load_sequence(video_or_files)

        feat = inferencer.extract_feature(seq)
        GALLERY_DB[subject_id] = feat

        return f"Successfully registered Subject <b>{subject_id}</b>! (Total Gallery Size: {len(GALLERY_DB)})"
    except Exception as e:
        return f"Error registering identity: {str(e)}"

def search_gallery(probe_file):
    """Searches a probe against registered gallery DB."""
    if not GALLERY_DB:
        return "Gallery is empty! Please register identities in the Registration tab first.", None

    if probe_file is None:
        return "Please upload a probe video or sequence!", None

    try:
        inferencer = get_inferencer()
        seq = load_sequence(probe_file)

        probe_feat = inferencer.extract_feature(seq)

        results = []
        for subject_id, gallery_feat in GALLERY_DB.items():
            mean_dist, cos_sim, match_score, _ = compute_metrics(probe_feat, gallery_feat)
            results.append((subject_id, mean_dist, cos_sim, match_score))

        results.sort(key=lambda x: x[1])

        table_rows = ""
        for rank, (sub_id, dist, cos_sim, score) in enumerate(results, 1):
            table_rows += f"""
            <tr>
                <td><b>#{rank}</b></td>
                <td>{sub_id}</td>
                <td>{dist:.4f}</td>
                <td>{cos_sim:.4f}</td>
                <td>{score:.1f}%</td>
            </tr>
            """

        result_html = f"""
        <table style="width:100%; border-collapse: collapse; text-align: left;">
            <thead>
                <tr style="border-bottom: 2px solid #555;">
                    <th>Rank</th><th>Subject ID</th><th>Euclidean Dist</th><th>Cosine Sim</th><th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """

        gif = create_preview_gif(seq)
        return result_html, gif
    except Exception as e:
        return f"Search Error: {str(e)}", None

def build_app():
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="OpenGait Demo Platform") as demo:
        gr.Markdown(
            """
            # OpenGait-2.0 Interactive Demo
            """
        )

        with gr.Tabs():
            with gr.TabItem("1-vs-1 Gait Verification"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Sequence A (Probe)")
                        probe_input = gr.File(label="Upload Probe Video (MP4/AVI) or Silhouette Images", file_count="multiple")
                        probe_preview = gr.Image(label="Probe Silhouette Preview", type="filepath")

                    with gr.Column():
                        gr.Markdown("### Sequence B (Gallery)")
                        gallery_input = gr.File(label="Upload Gallery Video (MP4/AVI) or Silhouette Images", file_count="multiple")
                        gallery_preview = gr.Image(label="Gallery Silhouette Preview", type="filepath")

                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, value=0.6, step=0.05,
                        label="Verification Cosine-Similarity Threshold (Higher = Stricter)"
                    )
                    verify_btn = gr.Button("Compare Gait Sequences", variant="primary", scale=2)

                verdict_output = gr.HTML(label="Verification Result")
                part_plot_output = gr.Plot(label="16-Part Feature Distance")

                verify_btn.click(
                    fn=verify_gait_pair,
                    inputs=[probe_input, gallery_input, threshold_slider],
                    outputs=[probe_preview, gallery_preview, verdict_output, part_plot_output]
                )

                # Pre-loaded sample video examples for quick testing
                sample_videos_dir = os.path.join(CURRENT_DIR, "videos")
                if os.path.exists(sample_videos_dir):
                    sample_files = sorted([os.path.join(sample_videos_dir, f) for f in os.listdir(sample_videos_dir) if f.endswith('.mp4')])
                    if len(sample_files) >= 2:
                        examples_list = []
                        for i in range(len(sample_files) - 1):
                            examples_list.append([[sample_files[i]], [sample_files[i+1]], 0.6])
                        gr.Examples(
                            examples=examples_list,
                            inputs=[probe_input, gallery_input, threshold_slider],
                            label="Sample Video Pairs for Quick Testing"
                        )

            with gr.TabItem("1-vs-N Identity Identification"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. Register Identities into Gallery DB")
                        reg_subject_id = gr.Textbox(label="Subject Identity / Name", placeholder="e.g. Subject_001")
                        reg_input = gr.File(label="Upload Subject Video / Silhouettes", file_count="multiple")
                        reg_btn = gr.Button("Register Subject into Gallery", variant="secondary")
                        reg_status = gr.HTML(value="Gallery empty.")

                    with gr.Column():
                        gr.Markdown("### 2. Query Probe against Gallery")
                        search_probe_input = gr.File(label="Upload Probe Query Video / Silhouettes", file_count="multiple")
                        search_btn = gr.Button("Search Gallery", variant="primary")
                        query_preview = gr.Image(label="Query Silhouette Preview", type="filepath")

                search_results_output = gr.HTML(label="Top Ranked Matches")

                reg_btn.click(
                    fn=register_identity,
                    inputs=[reg_subject_id, reg_input],
                    outputs=[reg_status]
                )

                search_btn.click(
                    fn=search_gallery,
                    inputs=[search_probe_input],
                    outputs=[search_results_output, query_preview]
                )

    return demo

if __name__ == "__main__":
    app = build_app()
    print("Launching Gradio app...")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
