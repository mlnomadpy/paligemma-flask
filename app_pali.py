import argparse
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" # set this in the env too
from flask import Flask, request, jsonify
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
# Set and check gpus

app = Flask(__name__)

model = None
processor = None
access_token = 'your HF token https://huggingface.co/settings/tokens' # don't hard cod it, use your env

def load_model_and_processor(model_path="google/paligemma-3b-mix-224", device="cuda:0", dtype=torch.bfloat16):
    global model
    global processor
    global access_token

    print("Loading the Model")
    if not processor:
        processor = AutoProcessor.from_pretrained(model_path, token=access_token)
    
    if not model:

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            token=access_token
        ).eval()
    print("Done loading the model...")
    return model, processor

def generate_image_caption(prompt, image_url, model, processor, max_new_tokens=20):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

@app.route('/image-caption', methods=['POST'])
def image_caption():
    data = request.json
    print(data)
    prompt = data['prompt']
    image_url = data['image_url']
    max_new_tokens = int(data.get('max_new_tokens', 20))

    try:
        model, processor = load_model_and_processor()
        caption = generate_image_caption(prompt, image_url, model, processor, max_new_tokens=max_new_tokens)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server with PaliGemma model for image captioning")
    args = parser.parse_args()
    print("Listening...")
    app.run(port=5011, host="0.0.0.0", debug=False)
