"""
Flask server host Qwen3-Instruct using HuggingFace transformers.

Duoc goi boi start_server.py

Ho tro tat ca cac model text cua Qwen3:

Qwen/Qwen3-8B-Instruct
Qwen/Qwen3-14B-Instruct
Qwen/Qwen3-32B-Instruct
Qwen/Qwen3-30B-A3B-Instruct
Qwen/Qwen3-235B-A22B-Instruct
"""

import os
import traceback

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B-Instruct")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"

print(f"Loading {MODEL_ID} ...")

quantization_config = None

# if LOAD_IN_4BIT:
#     from transformers import BitsAndBytesConfig
#
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
    quantization_config=quantization_config,
    trust_remote_code=True,
)

print(f"{MODEL_ID} ready!")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_ID,
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat():

    try:

        data = request.json

        messages = data.get("messages", [])

        max_new_tokens = data.get("max_tokens", 512)

        temperature = data.get("temperature", 0.0)

        do_sample = temperature > 0

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        del inputs, outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({

            "choices":[
                {
                    "message":{
                        "role":"assistant",
                        "content":generated_text
                    },
                    "finish_reason":"stop"
                }
            ],

            "model":MODEL_ID,

        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error":str(e)
        }),500


if __name__ == "__main__":

    port = int(os.environ.get("SERVER_PORT", 8000))

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=False,
    )