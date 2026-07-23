"""
Flask server host Qwen3 (Text) using HuggingFace transformers.

Compatible with OpenAI Chat Completions API.

Supported models:

Qwen/Qwen3-8B
Qwen/Qwen3-14B
Qwen/Qwen3-32B
Qwen/Qwen3-30B-A3B
Qwen/Qwen3-235B-A22B
"""

import os
import traceback

import torch
from flask import Flask, jsonify, request
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

app = Flask(__name__)

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Qwen/Qwen3-8B",
)

LOAD_IN_4BIT = os.environ.get(
    "LOAD_IN_4BIT",
    "0",
) == "1"

print(f"Loading {MODEL_ID} ...")

quantization_config = None

# if LOAD_IN_4BIT:
#
#     from transformers import BitsAndBytesConfig
#
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#     )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
    attn_implementation="sdpa",
)

print(f"{MODEL_ID} ready!")


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

        messages = data.get("messages", [])

        max_new_tokens = data.get("max_tokens", 512)

        temperature = float(
            data.get("temperature", 0)
        )

        top_p = float(
            data.get("top_p", 1.0)
        )

        do_sample = temperature > 0

        # ------------------------
        # HuggingFace official chat template
        # ------------------------

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        inputs.pop("token_type_ids", None)

        with torch.no_grad():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(
                inputs["input_ids"],
                generated_ids,
            )
        ]

        generated_text = tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        usage = {

            "prompt_tokens":
                int(inputs["input_ids"].shape[1]),

            "completion_tokens":
                int(generated_ids_trimmed[0].shape[0]),

            "total_tokens":
                int(inputs["input_ids"].shape[1])
                + int(generated_ids_trimmed[0].shape[0]),
        }

        del inputs
        del generated_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify(

            {

                "choices": [

                    {

                        "message": {

                            "role": "assistant",

                            "content": generated_text,

                        },

                        "finish_reason": "stop",

                    }

                ],

                "model": MODEL_ID,

                "usage": usage,

            }

        )

    except Exception as e:

        traceback.print_exc()

        return jsonify(

            {

                "error": str(e)

            }

        ), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "SERVER_PORT",
            8000,
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        threaded=False,

    )