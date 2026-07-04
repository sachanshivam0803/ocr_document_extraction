import os
from PIL import Image, ImageDraw, ImageFont

def generate_receipt():
    # Create a blank white image (width 400, height 500)
    image = Image.new("RGB", (400, 500), color="white")
    draw = ImageDraw.Draw(image)
    
    # Try using default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw SuperMart header
    draw.text((120, 40), "SuperMart Stores Inc.", fill="black", font=font)
    draw.text((100, 55), "100 Broadway, New York, NY", fill="gray", font=font)
    
    # Draw Date and Invoice number
    draw.text((50, 110), "Date: 2026-06-12", fill="black", font=font)
    draw.text((50, 140), "Invoice #: INV-2026-8941", fill="black", font=font)
    
    # Draw Divider
    draw.line((30, 175, 370, 175), fill="gray", width=1)
    
    # Items Headers
    draw.text((50, 185), "ITEM", fill="gray", font=font)
    draw.text((320, 185), "PRICE", fill="gray", font=font)
    
    # Items
    items = [
        ("Organic Milk 1 Gallon", "$5.99", 200),
        ("Whole Wheat Bread", "$3.49", 230),
        ("Fresh Bananas 1lb", "$1.89", 260),
        ("Greek Yogurt BlueBerry", "$4.50", 290),
        ("Detergent Liquid 50oz", "$12.99", 320)
    ]
    
    for name, price, y in items:
        draw.text((50, y), name, fill="black", font=font)
        draw.text((320, y), price, fill="black", font=font)
        
    # Draw Divider
    draw.line((30, 360, 370, 360), fill="gray", width=1)
    
    # Financials
    draw.text((200, 380), "Subtotal:", fill="black", font=font)
    draw.text((320, 380), "$28.87", fill="black", font=font)
    
    draw.text((200, 405), "Tax (8%):", fill="black", font=font)
    draw.text((320, 405), "$2.31", fill="black", font=font)
    
    # Draw double lines for total
    draw.line((200, 430, 370, 430), fill="black", width=1)
    draw.text((200, 440), "TOTAL:", fill="black", font=font)
    draw.text((310, 440), "$31.18", fill="black", font=font)
    draw.line((200, 465, 370, 465), fill="black", width=2)
    
    # Save image
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    image.save(os.path.join(samples_dir, "receipt_sample_1.jpg"), "JPEG")
    print("Generated receipt_sample_1.jpg successfully.")

def generate_invoice():
    # Create a blank white image (width 800, height 700)
    image = Image.new("RGB", (800, 700), color="white")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Company Logo and Header
    draw.rectangle((50, 45, 70, 85), fill="darkblue")
    draw.text((80, 50), "Apex Tech Solutions Ltd.", fill="black", font=font)
    draw.text((80, 65), "Enterprise Software Development & Consulting", fill="gray", font=font)
    
    # Invoice details block (top right)
    draw.text((550, 50), "INVOICE", fill="darkblue", font=font)
    draw.text((550, 90), "Invoice No: TX-90214", fill="black", font=font)
    draw.text((550, 120), "Date: 2026-05-28", fill="black", font=font)
    draw.text((550, 150), "Due Date: 2026-06-28", fill="black", font=font)
    
    # Bill To block
    draw.text((50, 150), "BILL TO:", fill="gray", font=font)
    draw.text((50, 170), "Global Retail Systems Corp", fill="black", font=font)
    draw.text((50, 185), "450 Enterprise Way, Suite 10", fill="black", font=font)
    draw.text((50, 200), "San Francisco, CA 94105", fill="black", font=font)
    
    # Header bar
    draw.rectangle((50, 275, 750, 305), fill="gray")
    draw.text((80, 285), "DESCRIPTION", fill="white", font=font)
    draw.text((620, 285), "AMOUNT", fill="white", font=font)
    
    # Items
    items = [
        ("Cloud Infrastructure Setup", "$1,200.00", 320),
        ("Database Migration Consulting", "$850.00", 360),
        ("Security Audit & Compliance", "$950.00", 400),
        ("Monthly Support Agreement", "$300.00", 440)
    ]
    
    for name, price, y in items:
        draw.text((80, y), name, fill="black", font=font)
        draw.text((620, y), price, fill="black", font=font)
        draw.line((50, y + 25, 750, y + 25), fill="lightgray", width=1)
        
    # Financial summary
    draw.text((500, 520), "Subtotal:", fill="black", font=font)
    draw.text((620, 520), "$3,300.00", fill="black", font=font)
    
    draw.text((500, 550), "VAT/Tax (8%):", fill="black", font=font)
    draw.text((620, 550), "$264.00", fill="black", font=font)
    
    # Total Box
    draw.rectangle((480, 580, 750, 630), fill="darkblue")
    draw.text((500, 600), "TOTAL DUE:", fill="white", font=font)
    draw.text((610, 600), "$3,564.00", fill="white", font=font)
    
    # Save image
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    image.save(os.path.join(samples_dir, "invoice_sample_2.jpg"), "JPEG")
    print("Generated invoice_sample_2.jpg successfully.")

if __name__ == "__main__":
    generate_receipt()
    generate_invoice()
