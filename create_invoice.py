from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (1200, 800), "white")
draw = ImageDraw.Draw(img)

font = ImageFont.load_default(size=28)

text = """INVOICE

Invoice Number: INV-001
Date: 07 August 2026

Vendor: ABC Technologies Pvt Ltd
Customer: XYZ Company

Item              Quantity    Price
Laptop            1           50000
Mouse             2           1000

Total Amount: 52000
"""

draw.multiline_text(
    (80, 60),
    text,
    fill="black",
    font=font,
    spacing=20
)

img.save(r"samples\invoice.png")

print("Invoice image created successfully!")