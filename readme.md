# PaliGemma Flask
This is the demo code for my talk in devmena
this readme will be improved later

don't forget to install the packages
pip install torch transformers flask accelerate bitsandbytes 

launch the server with 

python app_pali.py

and communicate with it from the terminal using 

curl -X POST http://localhost:5011/image-caption -H "Content-Type: application/json" -d '{
  "prompt": "caption en",
  "image_url": "https://huggingface.co/datasets/huggingface/d
ocumentation-images/resolve/main/transformers/tasks/car.jpg?d
ownload=true",
  "max_new_tokens": 100
}'
