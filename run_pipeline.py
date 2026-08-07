import json
import re
from pathlib import Path

INPUT = Path("samples/invoice.png")
OCR_JSON = Path("output/invoice_res.json")
MARKDOWN = Path("output/invoice.md")
SCHEMA = Path("configs/extraction_schema.json")
FINAL = Path("output/final_result.json")

data = json.loads(OCR_JSON.read_text(encoding="utf-8"))
schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
markdown = MARKDOWN.read_text(encoding="utf-8")

patterns = {
    "invoice_number": r"Invoice Number:\s*(.+)",
    "date": r"Date:\s*(.+)",
    "vendor": r"Vendor:\s*(.+)",
    "customer": r"Customer:\s*(.+)",
    "total_amount": r"Total Amount:\s*(.+)"
}

result = {}

for field, field_type in schema.items():
    match = re.search(patterns[field], markdown, re.IGNORECASE)

    if not match:
        result[field] = None
        continue

    value = match.group(1).strip()

    if field_type == "number":
        value = float(value.replace(",", ""))

    provenance = None

    for block in data["parsing_res_list"]:
        content = block.get("block_content", "")

        if re.search(patterns[field], content, re.IGNORECASE):
            provenance = {
                "page": 1,
                "source_text": content,
                "bbox": block.get("block_bbox"),
                "block_id": block.get("block_id"),
                "block_label": block.get("block_label")
            }
            break

    result[field] = {
        "value": value,
        "provenance": provenance
    }

FINAL.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("Pipeline completed successfully!")
print(json.dumps(result, indent=2))