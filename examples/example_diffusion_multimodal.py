from bforbuntyai import StableDiffusion, ImageCaptioner, auth

# HuggingFace login (required for some model versions)
auth.login()   # interactive, or: auth.login(token="hf_...")

# --- Stable Diffusion text-to-image ---
sd = StableDiffusion(model_id="runwayml/stable-diffusion-v1-5")
sd.generate(
    prompt="A futuristic city at sunset, 4K, photorealistic",
    negative_prompt="blurry, low quality, cartoon",
    n=4,
    steps=20,
    guidance_scale=7.5,
    save_dir="./sd_outputs",
)

# --- BLIP image captioning ---
cap = ImageCaptioner()
cap.caption("./sd_outputs/generated_0.png")

# Multiple images at once
cap.visualize([
    "./sd_outputs/generated_0.png",
    "./sd_outputs/generated_1.png",
    "./sd_outputs/generated_2.png",
])
