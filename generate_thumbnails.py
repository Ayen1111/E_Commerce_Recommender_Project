# generate_thumbnails.py
# Usage examples:
#   python generate_thumbnails.py --csv products.csv --out images --sample 2000 --update-csv
#   python generate_thumbnails.py --csv products.csv --out images --update-csv   (process all)
#
# Generates 400x300 PNG thumbnails from product names using Pillow.
# Writes products_with_images.csv with a new/updated 'image' column.

import os, csv, argparse, hashlib
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='products.csv', help='Input CSV path')
parser.add_argument('--out', default='images', help='Output image folder')
parser.add_argument('--name-col', default='product_name', help='Preferred product name column')
parser.add_argument('--image-col', default='image', help='Image column name to write')
parser.add_argument('--sample', type=int, default=2000, help='Rows to process (0 = all)')
parser.add_argument('--w', type=int, default=400)
parser.add_argument('--h', type=int, default=300)
parser.add_argument('--update-csv', action='store_true', help='Write products_with_images.csv')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# Try to load a sane font
def find_font():
    candidates = [
        "C:\\Windows\\Fonts\\SegoeUIBold.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_font()
if FONT_PATH:
    TITLE_FONT = ImageFont.truetype(FONT_PATH, 22)
    SUB_FONT   = ImageFont.truetype(FONT_PATH, 14)
else:
    TITLE_FONT = ImageFont.load_default()
    SUB_FONT   = ImageFont.load_default()

def short_name_and_hash(s, limit=28):
    s = ''.join(ch for ch in s if ch.isalnum() or ch in ' _-').strip()
    base = (s or 'product')[:limit].replace(' ', '_')
    h = hashlib.md5(s.encode('utf-8')).hexdigest()[:10]
    return f"{base}_{h}.png"

def draw_thumb(text, outpath, W=args.w, H=args.h):
    img = Image.new('RGB', (W, H), (20, 24, 28))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        shade = 22 + int(18 * (y / H))
        draw.line([(0, y), (W, y)], fill=(shade, shade + 2, shade + 4))

    title = (text or 'Product').strip()[:60]

    # ✅ Define text_size INSIDE draw_thumb so 'draw' is available
    def text_size(draw, text, font):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            return draw.textsize(text, font=font)

    # Draw the title
    tw, th = text_size(draw, title, TITLE_FONT)
    draw.text(
        ((W - tw) // 2, ((H - th) // 2) - 6),
        title,
        font=TITLE_FONT,
        fill=(255, 255, 255)
    )

    # Draw the subtext
    sub = "Preview"
    sw, sh = text_size(draw, sub, SUB_FONT)
    draw.text(
        ((W - sw) // 2, H - 34),
        sub,
        font=SUB_FONT,
        fill=(150, 170, 185)
    )

    img.save(outpath, optimize=True)


# Load CSV
with open(args.csv, newline='', encoding='utf-8', errors='ignore') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total = len(rows)
print(f"Found {total} rows in {args.csv}")

n = args.sample if args.sample and args.sample > 0 else total
n = min(n, total)
print(f"Generating thumbnails for {n} rows -> '{args.out}'")

# Resolve best name column present
def get_name(r):
    return (
        r.get(args.name_col) or r.get('product_name') or r.get('Product Name') or
        r.get('title') or r.get('name') or r.get('product title') or 'Product'
    )

updated = []
for i, r in enumerate(rows):
    if i < n:
        name = get_name(r)
        fname = short_name_and_hash(name)
        outp = os.path.join(args.out, fname)
        if not os.path.exists(outp):
            draw_thumb(name, outp)
        # write/overwrite image column to relative path
        r[args.image_col] = os.path.join(args.out, fname).replace('\\', '/')
    updated.append(r)

if args.update_csv:
    out_csv = 'products_with_images.csv'
    print("Writing:", out_csv)
    fieldnames = list(updated[0].keys())
    if args.image_col not in fieldnames:
        fieldnames.append(args.image_col)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(updated)

print("Done.")
