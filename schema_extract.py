import json
import re
from pathlib import Path

markdown_file = Path("output/invoice.md")
schema_file = Path("configs/extraction_schema.json")
output_file = Path("output/structured_output.json")

text = markdown_file.read_text(encoding="utf-8")
schema = json.loads(schema_file.read_text(encoding="utf-8"))

result = {}

patterns = {
    "invoice_number": r"Invoice Number:\s*(.+)",
    "date": r"Date:\s*(.+)",
    "vendor": r"Vendor:\s*(.+)",
    "customer": r"Customer:\s*(.+)",
    "total_amount": r"Total Amount:\s*([\d,]+)"
}

for field, field_type in schema.items():
    pattern = patterns.get(field)

    if pattern:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = match.group(1).strip()

            if field_type == "number":
                value = float(value.replace(",", ""))

            result[field] = value
        else:
            result[field] = None
    else:
        result[field] = None

output_file.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print("Schema extraction completed!")
print(json.dumps(result, indent=2))