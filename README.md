# Synthetic Data Generator (Open-Source LLM + Gradio)

Generate small synthetic datasets from a plain-English description + an explicit schema, using an open-source instruction-tuned LLM. Outputs:
- **JSON array** (list of row objects)
- **CSV download**
- **Data preview** in a Gradio UI

This project was built as part of an “open-source GenAI” learning track and is intentionally kept lightweight (not production-hardened).

---

## Why this exists

Synthetic data is useful when:
- real data is **scarce** or **expensive** to label
- data is **sensitive** (privacy / compliance)
- you want to **prototype** ML pipelines quickly
- you need **test fixtures** for downstream systems

This tool demonstrates a practical workflow:
**schema → prompt → generation → JSON validation/repair → CSV artifact**

---

## What it does

The user provides:
1. A dataset description (what the data represents)
2. A schema in `field:type` format (one per line)
3. A row count

The app:
1. Prompts the model to return **ONLY a JSON array**
2. Extracts the JSON array from the raw output
3. Validates:
   - output is a list of objects
   - no missing keys
   - no extra keys
4. If parsing fails, runs a **repair pass** (LLM-as-repair-tool)
5. Writes a CSV and makes it available for download

---

## Example schema format

```text
patient_id:str
age:int
sex:str
ef_percent:int
creatinine:float
readmitted_30d:bool
```

---

## Screenshots of UI
<img width="1508" height="713" alt="image" src="https://github.com/user-attachments/assets/2af45327-bf2c-4ee5-867e-a977d3104f7f" />
<img width="1508" height="713" alt="image" src="https://github.com/user-attachments/assets/5f3837f0-61a8-4d48-a427-0725517340fa" />
<img width="1508" height="713" alt="image" src="https://github.com/user-attachments/assets/af546a35-1ba6-4ef8-a31a-62c5bb60a73d" />


---

## Known Limitations
- Small row counts
- LLM output can be malformed
  - This is why we have a repair step call to an LLM
- Type checking is minimal
- Not production hardened
