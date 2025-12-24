import os
import json
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompts import SYSTEM_PROMPT, REPAIR_PROMPT


@dataclass
class ModelConfig:
    model_id: str = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9


def load_model(cfg: ModelConfig):
    """
    Loads a 4-bit quantized model if possible (bitsandbytes).
    Falls back to non-quantized if needed.
    """
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        # Optional: only needed for gated models
        login(token=hf_token)

    quant_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_quant_type = "nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Hitting the exception means we need more RAM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config = quant_config,
            device_map = "auto",
            torch_dtype = torch.float16
        )
    except Exception as e:
        print(f"WARNING: 4-bit quantized loading failed, falling back to full precision. Error: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            device_map = "auto",
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    model.eval()
    return tokenizer, model

# Tries to extract the JSON array from LLM output by []
def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a JSON array in the LLM output")
    return text[start : end + 1].strip()

# Parses the JSON array & looks for any errors
def _loads_json_array(text: str) -> List[Dict[str, Any]]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Parsed JSON isn't a list")
    if any(not isinstance(x, dict) for x in data):
        raise ValueError("JSON array doesn't contain objects")
    return data

# Basic Schema Validation (looks for missing keys or extra keys)
def _validate_schema(rows: List[Dict[str, Any]], fields: List[Tuple[str, str]]) -> None:
    expected_keys = [f for f, _t in fields]
    expected_set = set(expected_keys)

    for i, r in enumerate(rows):
        keys = set(r.keys())
        missing = expected_set - keys
        extra = keys - expected_set
        if missing:
            raise ValueError(f"Row {i} is missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"Row {i} has extra keys: {sorted(extra)}")

# Parses schema lines to return a list of (field, type)
def parse_schema(schema_text: str) -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
  
    for line in schema_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid schema line (expected field:type): '{line}'")
        
        # Just splitting the field name and its type
        field, type_str = line.split(":", 1)
        fields.append(
          (field.strip(), type_str.strip().lower())
        )

    if not fields:
        raise ValueError("No fields found in schema")
    return fields

# Uses tokenizer chat template - instruct models
def _chat_completion_hf(tokenizer, model, messages: List[Dict[str, str]], cfg: ModelConfig) -> str:
    # Prepare inputs for the model based on the current messages history
    enc = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Device Handling
    device = next(model.parameters()).device

    if isinstance(enc, torch.Tensor):
        if enc.dim() == 1:
            enc = enc.unsqueeze(0)
        input_ids = enc.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded


def generate_dataset_json(
    description: str,
    schema_text: str,
    num_rows: int,
    tokenizer,
    model,
    cfg: ModelConfig,
) -> Tuple[object, Optional[pd.DataFrame], Optional[str]]:
    try:
        fields = parse_schema(schema_text)
        if num_rows < 1 or num_rows > 200:
            raise ValueError("Number of rows must be between 1 and 200")

        schema_lines = "\n".join([f"- {f}: {t}" for f, t in fields])

        user_prompt = f"""Dataset description:
{description}

Schema (field:type):
{schema_lines}

Number of rows: {num_rows}

Return ONLY a JSON array of {num_rows} objects.
"""

        # Stateless messages per generation (so we don't overload memory)
        # We don't need to maintain the state since dataset generation should be stateless
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw = _chat_completion_hf(tokenizer, model, messages, cfg)
        candidate = _extract_json_array(raw)

        # Extract Rows of Data
        try:
            rows = _loads_json_array(candidate)
        except Exception:
            # Repair attempt using an LLM again!
            repair_messages = [
                {"role": "system", "content": REPAIR_PROMPT},
                {"role": "user", "content": candidate},
            ]
            repaired_raw = _chat_completion_hf(tokenizer, model, repair_messages, cfg)
            repaired_candidate = _extract_json_array(repaired_raw)
            rows = _loads_json_array(repaired_candidate)

        # Enforce Length
        if len(rows) != num_rows:
            rows = rows[:num_rows]

        # Validate the schema
        _validate_schema(rows, fields)

        df = pd.DataFrame(rows)

        # Write the CSV
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "synthetic_data.csv")
        df.to_csv(csv_path, index=False)

        return rows, df, csv_path

    except Exception as e:
        return {"Error": str(e)}, None, None
