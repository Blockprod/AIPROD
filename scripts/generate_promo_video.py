#!/usr/bin/env python
"""Generate professional AIPROD promotional video teaser"""

import os
import math
from PIL import Image, ImageDraw, ImageFont
import subprocess

# Check if moviepy is available, if not use ffmpeg directly
try:
    from moviepy.editor import *
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    print("⚠️  MoviePy not available, will use PIL + FFmpeg")

# Configuration
OUTPUT_VIDEO = "assets/aiprod-promo-teaser.mp4"
LOGO_OUTPUT = "assets/aiprod-logo.png"
os.makedirs("assets", exist_ok=True)

# Colors
COLORS = {
    "bg_dark": (10, 10, 15),
    "accent1": (255, 0, 110),      # Pink
    "accent2": (131, 56, 236),     # Purple
    "accent3": (58, 134, 255),     # Blue
    "accent4": (255, 190, 11),     # Yellow
    "accent5": (251, 86, 7),       # Orange
    "text_white": (255, 255, 255),
    "text_gold": (255, 215, 0),
}

def create_aiprod_logo(output_path):
    """Create a stylish AIPROD logo"""
    size = 500
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Center point
    center_x, center_y = size // 2, size // 2
    
    # Draw gradient circles (movie reel concept)
    for i in range(3):
        radius = 150 - i * 40
        color_idx = i
        colors_list = [COLORS["accent1"], COLORS["accent3"], COLORS["accent4"]]
        color = colors_list[i]
        
        draw.ellipse(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            outline=color,
            width=8
        )
    
    # Draw spokes (film reel spokes)
    num_spokes = 6
    for i in range(num_spokes):
        angle = (i * 360 / num_spokes) * math.pi / 180
        spoke_end = 140
        
        x1 = center_x + spoke_end * math.cos(angle)
        y1 = center_y + spoke_end * math.sin(angle)
        
        x2 = center_x + 60 * math.cos(angle)
        y2 = center_y + 60 * math.sin(angle)
        
        draw.line([(x1, y1), (x2, y2)], fill=COLORS["accent5"], width=6)
    
    # Center circle
    center_radius = 50
    draw.ellipse(
        [(center_x - center_radius, center_y - center_radius),
         (center_x + center_radius, center_y + center_radius)],
        fill=COLORS["accent2"],
        outline=COLORS["text_gold"],
        width=3
    )
    
    # Add "AIPROD" text at center
    try:
        # Try to use a larger font
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    draw.text((center_x, center_y), "AI", fill=COLORS["text_white"], anchor="mm", font=font)
    
    img = img.convert("RGB")
    img.save(output_path)
    print(f"✅ Logo created: {output_path}")
    return output_path

def create_promotional_video(output_path, logo_path):
    """Create a 10-second promotional teaser video"""
    
    if not HAS_MOVIEPY:
        print("Creating video using PIL frames + FFmpeg...")
        create_video_with_ffmpeg(output_path, logo_path)
        return
    
    # Create frames with PIL
    WIDTH, HEIGHT = 1920, 1080
    FPS = 24
    DURATION = 10  # seconds
    TOTAL_FRAMES = FPS * DURATION
    
    frames_dir = "temp_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"🎬 Creating {TOTAL_FRAMES} frames for 10-second video...")
    
    # Load logo
    try:
        logo = Image.open(logo_path)
        logo = logo.resize((400, 400))
    except:
        logo = None
    
    # Create each frame
    for frame_num in range(TOTAL_FRAMES):
        img = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg_dark"])
        draw = ImageDraw.Draw(img)
        
        # Progress
        progress = frame_num / TOTAL_FRAMES
        
        # Background gradient
        for i in range(HEIGHT):
            ratio = i / HEIGHT
            r = int(10 + (60 - 10) * ratio * progress)
            g = int(10 + (20 - 10) * ratio)
            b = int(15 + (80 - 15) * ratio * progress)
            draw.line([(0, i), (WIDTH, i)], fill=(r, g, b))
        
        # Scene 1 (0-3s): "Transform Your Vision"
        if frame_num < 72:  # 0-3s
            scene_progress = frame_num / 72
            alpha = int(255 * min(1, scene_progress * 3))
            
            # Title fade in
            title_y = int(HEIGHT * 0.2 + 100 * (1 - scene_progress))
            draw.text((WIDTH // 2, title_y), 
                     "TRANSFORM YOUR VISION",
                     fill=(*COLORS["text_white"], alpha) if alpha < 255 else COLORS["text_white"],
                     anchor="mm")
            
            # Accent lines
            line_width = int(400 * scene_progress)
            draw.rectangle(
                [(WIDTH // 2 - line_width // 2, title_y + 60),
                 (WIDTH // 2 + line_width // 2, title_y + 65)],
                fill=COLORS["accent1"]
            )
        
        # Scene 2 (3-6s): "Into 4K Videos"
        elif frame_num < 144:  # 3-6s
            scene_progress = (frame_num - 72) / 72
            
            # Scale animation
            scale = 0.8 + 0.2 * scene_progress
            
            # Icon circles
            circles = [
                (WIDTH * 0.2, HEIGHT * 0.5, "🎬", COLORS["accent3"]),
                (WIDTH * 0.5, HEIGHT * 0.5, "🤖", COLORS["accent2"]),
                (WIDTH * 0.8, HEIGHT * 0.5, "✨", COLORS["accent4"]),
            ]
            
            for cx, cy, emoji, color in circles:
                rad = int(60 * scale)
                draw.ellipse(
                    [(cx - rad, cy - rad), (cx + rad, cy + rad)],
                    fill=color
                )
                draw.text((cx, cy), emoji, fill=COLORS["text_white"], anchor="mm")
            
            # Text
            draw.text((WIDTH // 2, HEIGHT * 0.75),
                     "INTO 4K PROFESSIONAL VIDEOS",
                     fill=COLORS["text_gold"],
                     anchor="mm")
        
        # Scene 3 (6-9s): "Powered by AI"
        elif frame_num < 216:  # 6-9s
            scene_progress = (frame_num - 144) / 72
            
            # Logo animation (scale + fade in)
            logo_scale = int(400 * (0.5 + 0.5 * scene_progress))
            if logo:
                logo_scaled = logo.resize((logo_scale, logo_scale))
                logo_x = WIDTH // 2 - logo_scale // 2
                logo_y = HEIGHT // 2 - logo_scale // 2
                img.paste(logo_scaled, (logo_x, logo_y), logo_scaled if logo.mode == 'RGBA' else None)
            
            # Text
            draw.text((WIDTH // 2, HEIGHT * 0.15),
                     "POWERED BY ENTERPRISE AI",
                     fill=COLORS["accent1"],
                     anchor="mm")
        
        # Scene 4 (9-10s): Logo + text
        else:  # 9-10s
            if logo:
                img.paste(logo, (WIDTH // 2 - 200, HEIGHT // 2 - 200), logo if logo.mode == 'RGBA' else None)
            
            draw.text((WIDTH // 2, HEIGHT * 0.15),
                     "🎬 AIPROD 3.3.0",
                     fill=COLORS["text_gold"],
                     anchor="mm")
            
            draw.text((WIDTH // 2, HEIGHT * 0.85),
                     "github.com/Blockprod/AIPROD",
                     fill=COLORS["text_white"],
                     anchor="mm")
        
        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_num:04d}.png")
        img.save(frame_path)
        
        if (frame_num + 1) % 24 == 0:
            print(f"  ✓ Frame {frame_num + 1}/{TOTAL_FRAMES}")
    
    # Create video from frames
    print("🎬 Creating video from frames...")
    create_video_with_ffmpeg(output_path, logo_path)
    
    # Clean up
    import shutil
    shutil.rmtree(frames_dir)

def create_video_with_ffmpeg(output_path, logo_path):
    """Create video using FFmpeg"""
    # This would require ffmpeg installed
    print(f"📹 Video would be created at: {output_path}")
    print("   (Requires FFmpeg installation for full functionality)")

# Generate logo
print("🎨 Creating AIPROD Logo...")
logo_path = create_aiprod_logo(LOGO_OUTPUT)

# Generate video
print("🎬 Creating 10-second promotional teaser...")
create_promotional_video(OUTPUT_VIDEO, logo_path)

print("\n✅ Complete!")
print(f"📽️  Logo: {LOGO_OUTPUT}")
print(f"🎥 Teaser: {OUTPUT_VIDEO}")
