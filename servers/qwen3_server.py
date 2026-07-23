import os
import traceback
import torch

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B-Instruct")

print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
)

print("Ready.")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_ID,
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat():

    try:

        data = request.json

        messages = data["messages"]

        max_tokens = data.get("max_tokens", 512)

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
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return jsonify(
            {
                "choices":[
                    {
                        "message":{
                            "role":"assistant",
                            "content":answer
                        },
                        "finish_reason":"stop"
                    }
                ],
                "model":MODEL_ID
            }
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({"error":str(e)}),500


if __name__=="__main__":

    port=int(os.environ.get("SERVER_PORT",8000))

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=False
    )