import json
import re
from pathlib import Path

markdown_file = Path("output/invoice.md")
output_file = Path("output/provenance_output.json")

text = markdown_file.read_text(encoding="utf-8")

patterns = {
    "invoice_number": r"Invoice Number:\s*(.+)",
    "date": r"Date:\s*(.+)",
    "vendor": r"Vendor:\s*(.+)",
    "customer": r"Customer:\s*(.+)",
    "total_amount": r"Total Amount:\s*(.+)"
}

result = {}

for field, pattern in patterns.items():
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        value = match.group(1).strip()

        result[field] = {
            "value": value,
            "provenance": {
                "page": 1,
                "source_text": match.group(0)
            }
        }

output_file.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print("Provenance extraction completed!")
print(json.dumps(result, indent=2))