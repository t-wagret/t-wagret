from PIL import Image
import os

def optimize_image(input_path, output_dir, max_width=800):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    try:
        with Image.open(input_path) as img:
            # Resize
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                print(f"Resized to {max_width}x{new_height}")

            # Save as WebP
            webp_path = os.path.join(output_dir, "profile.webp")
            img.save(webp_path, "WEBP", quality=85)
            print(f"Saved {webp_path}")

            # Save as optimized JPEG
            jpg_path = os.path.join(output_dir, "profile_opt.jpg")
            img.convert("RGB").save(jpg_path, "JPEG", quality=85, optimize=True)
            print(f"Saved {jpg_path}")

            # AVIF might require external libraries or recent PILLOW, skipping if not available easily or using sips for it?
            # Pillow 10+ supports AVIF if libavif is installed. Let's try, if fails we catch it.
            try:
                avif_path = os.path.join(output_dir, "profile.avif")
                img.save(avif_path, "AVIF", quality=85)
                print(f"Saved {avif_path}")
            except Exception as e:
                print(f"AVIF processing failed (might need libavif): {e}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    optimize_image("assets/img/profile_large.jpg", "assets/img")
