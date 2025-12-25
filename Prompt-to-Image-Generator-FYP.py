import streamlit as st
import torch
import time
from io import BytesIO
from PIL import Image

# === Model Import: Stable Diffusion use kar rahe hain ===
from diffusers import StableDiffusionPipeline 

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
    st.session_state.show_welcome = True 

# ====== 3. Load Stable Diffusion Model ======
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"  # Aap yahan apna model path bhi de sakte hain
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Import aur Loading
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

# ====== 4. Sidebar ======
with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Inference Steps", 10, 50, 25) # Stable Diffusion ke liye use hoga
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5) 
    resolution = st.selectbox("Resolution", [256, 384, 512, 768], index=2)
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
<p style="font-size: 16px; margin-top: -5px;">Generate images from your text in a modern chat‑style experience</p>
<hr style="width: 50%; margin: 15px auto;">
<p style="font-size: 16px; font-weight: 500; margin-bottom: 10px;">This project is proudly built by:</p>
<p style="font-size: 18px; margin: 5px 0;"><b>Umar Farooq</b> — F21‑BSDS‑5008</p>
<p style="font-size: 18px; margin: 5px 0;"><b>Rao Bilal Zafar</b> — F21‑BSDS‑5024</p>
<p style="font-size: 18px; margin: 5px 0;"><b>Muhammad Shoaib Habib</b> — F21‑BSDS‑5029</p>
<hr style="width: 50%; margin: 15px auto;">
<p style="font-size: 15px; max-width: 500px; margin: auto;">
We extend our heartfelt gratitude to our professors, mentors, and supervisors for their invaluable guidance.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

# Show history
for msg in st.session_state.history:
    st.chat_message("user").write(msg["user"])
    with st.chat_message("assistant"):
        st.image(msg["image"], use_column_width=True, caption=f"Done in {msg['gen_time']:.2f}s")
        buf = BytesIO()
        msg["image"].save(buf, format="PNG")
        st.download_button("⬇️ Download", buf.getvalue(), file_name="generated.png", mime="image/png", key=f"dl_{hash(msg['user'] + str(msg['gen_time']))}")

# ====== Chat Input ======
user_prompt = st.chat_input("Type your image prompt…")

if user_prompt:
    st.session_state.show_welcome = False
    st.chat_message("user").write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Generating {resolution}×{resolution} image… please wait ⏳"):
            start_time = time.time()

            # Stable Diffusion Inference
            # Yahan humne steps aur guidance scale sidebar se uthaye hain
            image = pipe(
                user_prompt, 
                num_inference_steps=steps, 
                guidance_scale=guidance,
                height=resolution,
                width=resolution
            ).images[0]

            gen_time = time.time() - start_time

        st.image(image, use_column_width=True, caption=f"Done in {gen_time:.2f}s")
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button("⬇️ Download", buf.getvalue(), file_name="generated.png", mime="image/png", key=f"dl_current_{time.time()}")

    st.session_state.history.append({"user": user_prompt, "image": image, "gen_time": gen_time})

# source ~/miniconda3/bin/activate torchenv &&source ~/miniconda3/bin/activate torchenv && streamlit run /home/umarfarooq/Hands_on_Machine_Learning/Projects/Custom_image_Generation.py