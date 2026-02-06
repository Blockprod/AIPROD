#!/usr/bin/env python
"""Generate professional 10-second promotional teaser GIF"""

from PIL import Image, ImageDraw, ImageFont
import math
import os

OUTPUT_PATH = "assets/aiprod-teaser.gif"
os.makedirs("assets", exist_ok=True)

WIDTH, HEIGHT = 1920, 1080
FRAMES = 120  # 10 seconds at 12fps (optimized for GIF)

# Colors
COLORS = {
    "bg": (10, 10, 15),
    "accent1": (255, 0, 110),      # Pink
    "accent2": (131, 56, 236),     # Purple
    "accent3": (58, 134, 255),     # Blue
    "accent4": (255, 190, 11),     # Yellow
    "accent5": (251, 86, 7),       # Orange
    "text_white": (255, 255, 255),
    "text_gold": (255, 215, 0),
}

def create_frame(frame_num, total_frames):
    """Create a single promotional teaser frame"""
    img = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    
    # Calculate timings
    progress = frame_num / total_frames
    
    # Background gradient animation
    for i in range(HEIGHT):
        ratio = i / HEIGHT
        intensity = math.sin(progress * math.pi * 2) * 0.3 + 0.7
        r = int(10 + (80 - 10) * ratio * intensity)
        g = int(10 + (30 - 10) * ratio * intensity * 0.5)
        b = int(15 + (100 - 15) * ratio * intensity)
        draw.line([(0, i), (WIDTH, i)], fill=(r, g, b))
    
    # Scene 1 - TRANSFORM (0-2.5s / 0-30 frames)
    if frame_num < 30:
        scene_progress = frame_num / 30
        alpha_val = int(255 * min(1, scene_progress * 4))
        
        # Large title
        title_y = HEIGHT * 0.25
        title_text = "TRANSFORM"
        
        # Draw glowing effect
        for glow in range(5):
            glow_color = (
                int(COLORS["accent1"][0] * (1 - glow/5)),
                int(COLORS["accent1"][1] * (1 - glow/5)),
                int(COLORS["accent1"][2] * (1 - glow/5))
            )
            try:
                font = ImageFont.truetype("arial.ttf", 150)
                draw.text((WIDTH // 2, title_y), title_text, 
                         fill=glow_color, anchor="mm", font=font)
            except:
                draw.text((WIDTH // 2, title_y), title_text, 
                         fill=glow_color, anchor="mm")
        
        # Accent line
        line_width = int(500 * scene_progress)
        draw.rectangle(
            [(WIDTH // 2 - line_width // 2, title_y + 100),
             (WIDTH // 2 + line_width // 2, title_y + 110)],
            fill=COLORS["accent1"]
        )
    
    # Scene 2 - ICONS (2.5-5s / 30-60 frames)
    elif frame_num < 60:
        scene_progress = (frame_num - 30) / 30
        
        # Three emoji circles with scale animation
        icons = ["🎬", "🤖", "✨"]
        positions = [WIDTH * 0.2, WIDTH * 0.5, WIDTH * 0.8]
        colors_list = [COLORS["accent3"], COLORS["accent2"], COLORS["accent4"]]
        
        for idx, (emoji, x, color) in enumerate(zip(icons, positions, colors_list)):
            # Staggered scale
            stagger = scene_progress - idx * 0.15
            if stagger > 0:
                scale = 0.3 + 0.7 * min(1, stagger)
                radius = int(80 * scale)
                y = HEIGHT * 0.5
                
                # Draw circle
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    fill=color
                )
                
                # Draw emoji
                emoji_size = int(150 * scale)
                try:
                    emoji_font = ImageFont.truetype("arial.ttf", emoji_size)
                    draw.text((x, y), emoji, fill=COLORS["text_white"],
                             anchor="mm", font=emoji_font)
                except:
                    draw.text((x, y), emoji, fill=COLORS["text_white"],
                             anchor="mm")
        
        # Text below
        try:
            text_font = ImageFont.truetype("arial.ttf", 80)
            draw.text((WIDTH // 2, HEIGHT * 0.85), "INTO 4K VIDEOS",
                     fill=COLORS["text_gold"], anchor="mm", font=text_font)
        except:
            draw.text((WIDTH // 2, HEIGHT * 0.85), "INTO 4K VIDEOS",
                     fill=COLORS["text_gold"], anchor="mm")
    
    # Scene 3 - AI POWERED (5-8s / 60-96 frames)
    elif frame_num < 96:
        scene_progress = (frame_num - 60) / 36
        
        # Spinning star effect
        angle = progress * 360 * 4
        angle_rad = math.radians(angle)
        
        center_x, center_y = WIDTH // 2, HEIGHT * 0.35
        radius = 200
        
        # Draw rotating circles
        for i in range(3):
            r = radius * (1 - i * 0.25)
            x = center_x + r * math.cos(angle_rad + i * math.pi * 2 / 3)
            y = center_y + r * math.sin(angle_rad + i * math.pi * 2 / 3)
            
            colors_cycle = [COLORS["accent2"], COLORS["accent3"], COLORS["accent5"]]
            draw.ellipse(
                [(x - 40, y - 40), (x + 40, y + 40)],
                fill=colors_cycle[i]
            )
        
        # Text
        try:
            text_font = ImageFont.truetype("arial.ttf", 90)
            draw.text((WIDTH // 2, HEIGHT * 0.75), "POWERED BY AI",
                     fill=COLORS["accent1"], anchor="mm", font=text_font)
        except:
            draw.text((WIDTH // 2, HEIGHT * 0.75), "POWERED BY AI",
                     fill=COLORS["accent1"], anchor="mm")
    
    # Scene 4 - LOGO & CTA (8-10s / 96-120 frames)
    else:
        scene_progress = (frame_num - 96) / 24
        
        # Load and display logo if available
        try:
            logo = Image.open("assets/aiprod-logo.png")
            logo_size = int(300 * (0.8 + 0.2 * scene_progress))
            logo = logo.resize((logo_size, logo_size))
            logo_x = WIDTH // 2 - logo_size // 2
            logo_y = HEIGHT // 2 - logo_size // 2 - 100
            img.paste(logo, (logo_x, logo_y), logo if logo.mode == 'RGBA' else None)
        except:
            pass
        
        # Brand name
        try:
            text_font = ImageFont.truetype("arial.ttf", 100)
            draw.text((WIDTH // 2, HEIGHT * 0.15), "AIPROD 3.3.0",
                     fill=COLORS["text_gold"], anchor="mm", font=text_font)
            
            # CTA
            draw.text((WIDTH // 2, HEIGHT * 0.9),
                     "github.com/Blockprod/AIPROD",
                     fill=COLORS["text_white"], anchor="mm", font=text_font)
        except:
            draw.text((WIDTH // 2, HEIGHT * 0.15), "AIPROD 3.3.0",
                     fill=COLORS["text_gold"], anchor="mm")
            draw.text((WIDTH // 2, HEIGHT * 0.9),
                     "github.com/Blockprod/AIPROD",
                     fill=COLORS["text_white"], anchor="mm")
    
    return img

# Generate all frames
print("🎬 Generating 10-second promotional teaser GIF...")
frames = []

for frame_num in range(FRAMES):
    frame = create_frame(frame_num, FRAMES)
    frames.append(frame)
    
    if (frame_num + 1) % 24 == 0:
        seconds = (frame_num + 1) / 12
        print(f"  ✓ Frame {frame_num + 1}/{FRAMES} ({seconds:.1f}s)")

# Save as GIF
frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=83,  # ~10 seconds total (120 frames * 83ms)
    loop=0,
    optimize=False
)

print(f"\n✅ Promotional teaser created: {OUTPUT_PATH}")
print(f"📊 Dimensions: {WIDTH}x{HEIGHT}px")
print(f"🎬 Frames: {len(frames)}")
print(f"⏱️ Duration: 10 seconds")
print(f"🎨 Scenes: Transform → Icons → AI → Logo+CTA")
