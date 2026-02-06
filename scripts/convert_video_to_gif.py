"""Vérifier les outils et convertir la vidéo en GIF pour le README."""
import os
import shutil

path = os.path.join("assets", "promo", "aiprod_promo.mp4")
size = os.path.getsize(path)
print(f"Video: {path}")
print(f"Taille: {size / 1024:.0f} KB ({size / 1024 / 1024:.1f} MB)")

ffmpeg = shutil.which("ffmpeg")
if ffmpeg:
    print(f"ffmpeg: {ffmpeg}")
else:
    print("ffmpeg: NON INSTALLE")

# Vérifier les packages Python
for pkg in ["moviepy", "imageio", "PIL", "cv2"]:
    try:
        __import__(pkg)
        print(f"{pkg}: INSTALLE")
    except ImportError:
        print(f"{pkg}: PAS INSTALLE")

# --- Conversion MP4 -> GIF ---
gif_path = os.path.join("assets", "promo", "aiprod_promo.gif")

if ffmpeg:
    print("\n=== Conversion MP4 -> GIF avec ffmpeg ===")
    # Créer un GIF optimisé: 15fps, largeur 800px, max 10MB
    cmd = (
        f'"{ffmpeg}" -i "{path}" '
        f'-vf "fps=12,scale=800:-1:flags=lanczos,split[s0][s1];'
        f'[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" '
        f'-loop 0 "{gif_path}" -y'
    )
    print(f"  Commande: {cmd[:100]}...")
    result = os.system(cmd)
    
    if os.path.exists(gif_path):
        gif_size = os.path.getsize(gif_path)
        print(f"  GIF genere: {gif_path}")
        print(f"  Taille: {gif_size / 1024:.0f} KB ({gif_size / 1024 / 1024:.1f} MB)")
    else:
        print("  Echec de la conversion ffmpeg")
else:
    # Fallback: utiliser cv2 + Pillow
    print("\n=== Conversion MP4 -> GIF avec cv2 + Pillow ===")
    try:
        import cv2
        from PIL import Image
        
        cap = cv2.VideoCapture(path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  FPS: {fps}, Total frames: {total_frames}")
        
        # Prendre 1 frame sur 2 pour réduire la taille
        skip = max(1, int(fps / 12))  # Cible ~12fps
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % skip == 0:
                # Redimensionner à 800px de large
                h, w = frame.shape[:2]
                new_w = 800
                new_h = int(h * new_w / w)
                frame = cv2.resize(frame, (new_w, new_h))
                # Convertir BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            count += 1
        
        cap.release()
        print(f"  Frames extraites: {len(frames)}")
        
        if frames:
            # Sauvegarder en GIF
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / 12),  # ~12fps
                loop=0,
                optimize=True,
            )
            gif_size = os.path.getsize(gif_path)
            print(f"  GIF genere: {gif_path}")
            print(f"  Taille: {gif_size / 1024:.0f} KB ({gif_size / 1024 / 1024:.1f} MB)")
            
    except ImportError as e:
        print(f"  Erreur import: {e}")
        print("  Installation des packages necessaires...")

print("\n=== Fichiers finaux ===")
promo_dir = os.path.join("assets", "promo")
for f in sorted(os.listdir(promo_dir)):
    fpath = os.path.join(promo_dir, f)
    size = os.path.getsize(fpath)
    print(f"  {f}: {size / 1024:.0f} KB")
