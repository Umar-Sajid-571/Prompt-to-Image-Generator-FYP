import streamlit as st
import torch
import torch.nn.functional as F
import time
from io import BytesIO
from PIL import Image

# === Import your GAN Generator ===
from model import ConditionalGenerator

# ====== 1. Page + Theme Config ======
st.set_page_config(
    page_title="Prompt to Image Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      :root { color-scheme: light dark; }
      .footer { text-align:center; font-size:12px; color:gray; margin-top:2em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====== 2. Initialize Session State ======
if "history" not in st.session_state:
    st.session_state.history = []
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True  # Show credits on first load

# ====== Your class order from training ======
CLASS_ORDER = ["cat", "dog", "car", "tree"]  # <-- change to your training classes
LABEL_MAP = {name.lower(): idx for idx, name in enumerate(CLASS_ORDER)}


# ====== 3. Load Custom Generator Model ======
@st.cache_resource
def load_model():
    noise_dim = 384
    embed_dim = 384
    num_classes = len(CLASS_ORDER)
    img_channels = 3
    feature_maps = 64

    G = ConditionalGenerator(
        noise_dim, embed_dim, num_classes, img_channels, feature_maps
    )
    G.load_state_dict(torch.load("image_generator_model.pt", map_location="cpu"))
    G.eval()
    return G, noise_dim, num_classes


G, z_dim, num_classes = load_model()

# ====== 4. Sidebar ======
with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Inference Steps", 10, 50, 18)  # Not used in GAN
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)  # Placeholder
    resolution = st.selectbox("Resolution", [64, 128, 256, 384, 448, 512], index=0)
    st.markdown("---")
    st.markdown(
        "<div class='footer'>© 2025 UBS_developers</div>", unsafe_allow_html=True
    )

# ====== 5. Main: Chat UI ======
st.title("🎨 Custom Image Generator")
st.caption("Generate images by your text in modern chat flow.")

# Show Welcome / Credits Message
if st.session_state.show_welcome:
    with st.chat_message("assistant"):
        st.markdown(
            """
<div style="text-align: center;">

<h2>👋 Welcome to the Custom Image Generator</h2>
<p style="font-size: 16px; margin-top: -5px;">
Generate images from your text in a modern chat‑style experience
</p>

<hr style="width: 50%; margin: 15px auto;">

<p style="font-size: 16px; font-weight: 500; margin-bottom: 10px;">
This project is proudly built by:
</p>

<p style="font-size: 18px; margin: 5px 0;"><b>Umar Farooq</b> — F21‑BSDS‑5008</p>
<p style="font-size: 18px; margin: 5px 0;"><b>Rao Bilal Zafar</b> — F21‑BSDS‑5024</p>
<p style="font-size: 18px; margin: 5px 0;"><b>Muhammad Shoaib Habib</b> — F21‑BSDS‑5029</p>

<hr style="width: 50%; margin: 15px auto;">

<p style="font-size: 15px; max-width: 500px; margin: auto;">
We extend our heartfelt gratitude to our professors, mentors, and supervisors  
for their invaluable guidance, encouragement, and support throughout this journey💖💖.
</p>

<p style="font-size: 14px; color: gray; margin-top: 15px;">
💡 Tip: Type your first prompt below to begin.<br>
On custom settings, generation may take a few seconds — quality takes time.
</p>

</div>
""",
            unsafe_allow_html=True,
        )

# Show history
for msg in st.session_state.history:
    st.chat_message("user").write(msg["user"])
    with st.chat_message("assistant"):
        st.image(
            msg["image"],
            use_column_width=True,
            caption=f"Done in {msg['gen_time']:.2f}s",
        )
        buf = BytesIO()
        msg["image"].save(buf, format="PNG")
        st.download_button(
            "⬇️ Download",
            buf.getvalue(),
            file_name="generated.png",
            mime="image/png",
            key=f"dl_{hash(msg['user'] + str(msg['gen_time']))}",
        )

# ====== Chat Input ======
user_prompt = st.chat_input("Type your image prompt…")

if user_prompt:
    st.session_state.show_welcome = False
    st.chat_message("user").write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Generating {resolution}×{resolution} image… please wait ⏳"):
            start_time = time.time()

            # Map prompt to class id
            class_id = LABEL_MAP.get(user_prompt.lower(), 0)

            # Noise + label
            z = torch.randn(1, z_dim)
            label = torch.tensor([class_id], dtype=torch.long)

            with torch.no_grad():
                out = G(z, label)[0]  # (3, H, W) in [-1,1]
                img = ((out + 1) / 2).clamp(0, 1)  # to [0,1]
                img = img.mul(255).byte().permute(1, 2, 0).numpy()

                # Resize to user‑selected resolution
                image = Image.fromarray(img).resize(
                    (resolution, resolution), Image.BICUBIC
                )
                gen_time = time.time() - start_time

        st.image(image, use_column_width=True, caption=f"Done in {gen_time:.2f}s")
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download",
            buf.getvalue(),
            file_name="generated.png",
            mime="image/png",
            key=f"dl_current_{time.time()}",
        )

    st.session_state.history.append(
        {"user": user_prompt, "image": image, "gen_time": gen_time}
    )
