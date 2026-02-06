#!/usr/bin/env python
"""Generate advanced animated GIF banner for AIPROD README"""

from PIL import Image, ImageDraw
import math
import os

# Configuration
WIDTH = 1920
HEIGHT = 380
FRAMES = 120
OUTPUT_PATH = "assets/banner-animation.gif"

os.makedirs("assets", exist_ok=True)

# Advanced color palette
COLORS = {
    "bg_dark": (10, 10, 10),
    "bg_light": (20, 25, 35),
    "accent1": (255, 0, 110),      # Pink
    "accent2": (131, 56, 236),     # Purple
    "accent3": (58, 134, 255),     # Blue
    "accent4": (255, 190, 11),     # Yellow
    "accent5": (251, 86, 7),       # Orange
    "text_white": (255, 255, 255),
    "text_gold": (255, 190, 11),
    "border": (74, 144, 226),
}

STAGE_COLORS = [COLORS["accent1"], COLORS["accent2"], COLORS["accent3"], COLORS["accent4"], COLORS["accent5"]]
STAGE_EMOJIS = ["📝", "🧠", "🎬", "✨", "🚀"]
STAGE_LABELS = ["SCRIPTS", "INTELLIGENCE", "CREATION", "EXCELLENCE", "LAUNCH"]

def draw_gradient_circle(draw, x, y, radius, color, intensity=1.0):
    """Draw a circle with glow effect"""
    # Outer glow
    for r in range(radius + 15, radius, -2):
        glow_alpha = int(255 * (1 - (r - radius) / 15) * intensity * 0.3)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], 
                    outline=color + (glow_alpha,) if len(color) == 3 else color,
                    width=2)
    
    # Main circle
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                fill=color)

def create_frame(frame_num, total_frames, width, height):
    """Create a single frame of the animation"""
    img = Image.new("RGB", (width, height), COLORS["bg_dark"])
    draw = ImageDraw.Draw(img)
    
    # Calculate animation progression
    progress = (frame_num % total_frames) / total_frames
    cycle_progress = math.sin(progress * math.pi * 2) * 0.5 + 0.5  # Smooth sine wave
    
    # Draw gradient background effect
    for i in range(height):
        ratio = i / height
        r = int(10 + (20 - 10) * ratio)
        g = int(10 + (25 - 10) * ratio)
        b = int(10 + (35 - 10) * ratio)
        draw.line([(0, i), (width, i)], fill=(r, g, b))
    
    # Draw top border glow
    for i in range(10):
        alpha = int(255 * (1 - i / 10) * 0.3)
        border_color = COLORS["border"]
        draw.line([(0, i), (width, i)], fill=border_color)
    
    # Draw border
    border_width = 2
    draw.rectangle(
        [(border_width, border_width), (width - border_width, height - border_width)],
        outline=COLORS["border"],
        width=border_width
    )
    
    # Title and subtitle
    title_y = 35
    draw.text((width // 2, title_y), "🎬 AIPROD - AI VIDEO TRANSFORMATION", 
              fill=COLORS["text_white"], anchor="mm")
    
    subtitle_y = 75
    draw.text((width // 2, subtitle_y), 
              "Transform Your Creative Vision Into Professional 4K Videos",
              fill=COLORS["text_gold"], anchor="mm")
    
    # Stage positions with more spacing
    stage_positions = [250, 650, 1050, 1450, 1800]
    stage_y = 160
    base_radius = 45
    
    # Draw connection lines with animation
    for idx in range(len(stage_positions) - 1):
        x1 = stage_positions[idx]
        x2 = stage_positions[idx + 1]
        
        # Line animation
        line_progress = (progress * len(stage_positions) - idx) * 2
        line_progress = max(0, min(1, line_progress))
        
        if line_progress > 0:
            x_end = x1 + (x2 - x1) * line_progress
            draw.line([(x1 + base_radius + 15, stage_y), (x_end, stage_y)],
                     fill=STAGE_COLORS[idx], width=5)
            
            if line_progress >= 0.95:
                # Arrow head
                arrow_size = 12
                draw.polygon([(x_end, stage_y),
                            (x_end - arrow_size, stage_y - arrow_size),
                            (x_end - arrow_size, stage_y + arrow_size)],
                           fill=STAGE_COLORS[idx])
    
    # Draw stages with staggered animation
    for idx, x in enumerate(stage_positions):
        stage_progress = (progress * len(stage_positions) - idx) / 2
        stage_progress = max(0, min(1, stage_progress))
        
        if stage_progress > 0:
            # Calculate pulsing effect
            pulse = math.sin(frame_num * 0.15 + idx) * 5
            current_radius = base_radius + pulse * stage_progress
            
            # Draw circle with glow
            draw.ellipse(
                [(x - current_radius, stage_y - current_radius), 
                 (x + current_radius, stage_y + current_radius)],
                fill=STAGE_COLORS[idx]
            )
            
            # Glow ring
            for ring in range(3):
                ring_radius = current_radius + (ring + 1) * 3
                ring_intensity = (1 - ring / 3) * stage_progress * 0.4
                draw.ellipse(
                    [(x - ring_radius, stage_y - ring_radius), 
                     (x + ring_radius, stage_y + ring_radius)],
                    outline=STAGE_COLORS[idx],
                    width=2
                )
            
            # Draw emoji
            draw.text((x, stage_y - 12), STAGE_EMOJIS[idx], 
                     fill=COLORS["text_white"], anchor="mm")
            
            # Draw label
            draw.text((x, stage_y + 35), STAGE_LABELS[idx],
                     fill=COLORS["text_white"], anchor="mm")
    
    # Draw animated progress bar
    progress_bar_y = height - 50
    progress_bar_width = width - 100
    bar_height = 8
    
    # Background bar with gradient
    draw.rectangle(
        [(50, progress_bar_y), (50 + progress_bar_width, progress_bar_y + bar_height)],
        outline=COLORS["border"],
        width=2
    )
    
    # Animated fill with gradient
    fill_width = progress_bar_width * progress
    segment_width = 20
    
    for i in range(int(fill_width / segment_width) + 1):
        segment_x = 50 + i * segment_width
        segment_end = min(segment_x + segment_width, 50 + fill_width)
        
        # Gradient color based on position
        color_ratio = i / (progress_bar_width / segment_width)
        if color_ratio < 0.25:
            color = COLORS["accent1"]
        elif color_ratio < 0.5:
            color = COLORS["accent2"]
        elif color_ratio < 0.75:
            color = COLORS["accent3"]
        elif color_ratio < 0.9:
            color = COLORS["accent4"]
        else:
            color = COLORS["accent5"]
        
        if segment_end > segment_x:
            draw.rectangle(
                [(segment_x, progress_bar_y), (segment_end, progress_bar_y + bar_height)],
                fill=color
            )
    
    # Percentage text
    percentage = int(progress * 100)
    draw.text((width // 2, progress_bar_y + bar_height + 20),
             f"Pipeline Execution: {percentage}%",
             fill=COLORS["text_white"], anchor="mm")
    
    return img

# Generate all frames
print("🎬 Generating advanced animated banner...")
frames = []

for frame_num in range(FRAMES):
    frame = create_frame(frame_num, FRAMES, WIDTH, HEIGHT)
    frames.append(frame)
    
    if (frame_num + 1) % 20 == 0:
        print(f"  ✓ Frame {frame_num + 1}/{FRAMES}")

# Save as GIF
frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=40,  # 40ms per frame for smooth animation
    loop=0
)

print(f"\n✅ Advanced GIF banner created: {OUTPUT_PATH}")
print(f"📊 Dimensions: {WIDTH}x{HEIGHT}px")
print(f"🎬 Frames: {len(frames)}")
print(f"⏱️ Duration: {len(frames) * 40 / 1000:.1f} seconds per loop")
print(f"🎨 Features: Gradient backgrounds, pulsing effects, staggered animations, glow rings")
