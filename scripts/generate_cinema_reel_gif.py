#!/usr/bin/env python
"""Generate cinema film reel animated GIF banner for AIPROD README"""

from PIL import Image, ImageDraw
import math
import os

# Configuration
WIDTH = 1920
HEIGHT = 400
FRAMES = 180
OUTPUT_PATH = "assets/banner-animation.gif"

os.makedirs("assets", exist_ok=True)

# Color palette - Vintage cinema style
COLORS = {
    "bg": (15, 15, 20),
    "bg_light": (25, 25, 35),
    "reel": (230, 190, 80),  # Golden
    "reel_dark": (140, 110, 40),  # Dark gold
    "film": (20, 20, 25),  # Very dark
    "accent1": (255, 0, 110),      # Pink
    "accent2": (131, 56, 236),     # Purple
    "accent3": (58, 134, 255),     # Blue
    "accent4": (255, 190, 11),     # Yellow
    "accent5": (251, 86, 7),       # Orange
    "text_white": (255, 255, 255),
    "text_gold": (255, 215, 0),
    "spotlight": (255, 255, 200),
}

STAGE_COLORS = [COLORS["accent1"], COLORS["accent2"], COLORS["accent3"], COLORS["accent4"], COLORS["accent5"]]
STAGE_LABELS = ["SCRIPTS", "INTELLIGENCE", "CREATION", "EXCELLENCE", "LAUNCH"]

def draw_reel_hole(draw, cx, cy, size, frame_angle):
    """Draw rotated film reel holes"""
    num_holes = 12
    hole_radius = 8
    
    for i in range(num_holes):
        angle = (i * 360 / num_holes + frame_angle) % 360
        angle_rad = math.radians(angle)
        
        hole_x = cx + size * math.cos(angle_rad)
        hole_y = cy + size * math.sin(angle_rad)
        
        draw.ellipse(
            [(hole_x - hole_radius, hole_y - hole_radius),
             (hole_x + hole_radius, hole_y + hole_radius)],
            fill=COLORS["bg"]
        )

def draw_film_reel(draw, cx, cy, outer_radius, frame_angle, rotation_speed=2):
    """Draw a realistic film reel with rotation"""
    
    actual_angle = (frame_angle * rotation_speed) % 360
    
    # Outer ring
    draw.ellipse(
        [(cx - outer_radius, cy - outer_radius),
         (cx + outer_radius, cy + outer_radius)],
        fill=COLORS["reel"],
        outline=COLORS["reel_dark"],
        width=5
    )
    
    # Inner ring darker
    inner_radius = outer_radius * 0.75
    draw.ellipse(
        [(cx - inner_radius, cy - inner_radius),
         (cx + inner_radius, cy + inner_radius)],
        fill=COLORS["reel_dark"]
    )
    
    # Draw holes pattern
    draw_reel_hole(draw, cx, cy, outer_radius * 0.5, actual_angle)
    draw_reel_hole(draw, cx, cy, outer_radius * 0.7, actual_angle + 30)
    
    # Center hub
    hub_radius = outer_radius * 0.3
    draw.ellipse(
        [(cx - hub_radius, cy - hub_radius),
         (cx + hub_radius, cy + hub_radius)],
        fill=COLORS["reel_dark"]
    )
    
    # Spokes
    num_spokes = 6
    for i in range(num_spokes):
        angle = (i * 360 / num_spokes + actual_angle) % 360
        angle_rad = math.radians(angle)
        
        spoke_start_x = cx + hub_radius * math.cos(angle_rad)
        spoke_start_y = cy + hub_radius * math.sin(angle_rad)
        spoke_end_x = cx + inner_radius * 0.95 * math.cos(angle_rad)
        spoke_end_y = cy + inner_radius * 0.95 * math.sin(angle_rad)
        
        draw.line(
            [(spoke_start_x, spoke_start_y), (spoke_end_x, spoke_end_y)],
            fill=COLORS["reel"],
            width=3
        )

def create_frame(frame_num, total_frames, width, height):
    """Create a single frame of the film reel animation"""
    img = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    
    # Background gradient
    for i in range(height):
        ratio = i / height
        r = int(15 + (25 - 15) * ratio)
        g = int(15 + (25 - 15) * ratio)
        b = int(20 + (35 - 20) * ratio)
        draw.line([(0, i), (width, i)], fill=(r, g, b))
    
    # Decorative border
    border_width = 2
    draw.rectangle(
        [(border_width, border_width), (width - border_width, height - border_width)],
        outline=COLORS["text_gold"],
        width=border_width
    )
    
    # Title - Cinema style
    draw.text((width // 2, 35), "📽️ AIPROD CINEMA TRANSFORMATION 📽️",
              fill=COLORS["text_gold"], anchor="mm")
    
    # Left reel (coming from)
    left_reel_x = 300
    reel_y = 140
    reel_radius = 70
    
    progress = (frame_num % total_frames) / total_frames
    
    draw_film_reel(draw, left_reel_x, reel_y, reel_radius, frame_num)
    
    # Film strip coming from left reel
    film_start_y = reel_y - 15
    film_height = 30
    film_spacing = 8
    
    # Left side films (input)
    for frame_idx in range(3):
        film_x = left_reel_x + reel_radius + 30 + frame_idx * (60 + film_spacing)
        # Draw film frame
        draw.rectangle(
            [(film_x, film_start_y), (film_x + 60, film_start_y + film_height)],
            fill=COLORS["film"],
            outline=COLORS["text_gold"],
            width=1
        )
        # Sprocket holes
        for hole_idx in range(4):
            hole_y_pos = film_start_y - 8 - hole_idx * 12
            hole_y_pos_bottom = film_start_y + film_height + 8 + hole_idx * 12
            
            draw.rectangle(
                [(film_x + 5, hole_y_pos), (film_x + 15, hole_y_pos + 5)],
                fill=COLORS["text_gold"]
            )
            draw.rectangle(
                [(film_x + 45, hole_y_pos), (film_x + 55, hole_y_pos + 5)],
                fill=COLORS["text_gold"]
            )
            draw.rectangle(
                [(film_x + 5, hole_y_pos_bottom), (film_x + 15, hole_y_pos_bottom + 5)],
                fill=COLORS["text_gold"]
            )
            draw.rectangle(
                [(film_x + 45, hole_y_pos_bottom), (film_x + 55, hole_y_pos_bottom + 5)],
                fill=COLORS["text_gold"]
            )
        
        # Stage label
        stage_idx = frame_idx
        if stage_idx < 2:
            draw.text((film_x + 30, film_start_y + film_height + 25),
                     STAGE_LABELS[stage_idx],
                     fill=STAGE_COLORS[stage_idx % len(STAGE_COLORS)], anchor="mm")
    
    # Processing stages in the middle with transformation effect
    processing_y = 200
    stage_spacing = 280
    
    for stage_idx in range(3):
        stage_x = 400 + stage_idx * stage_spacing
        
        # Calculate animation progress for this stage
        stage_progress = (progress * 5 - stage_idx) % 1
        
        if stage_idx == 0:  # Processing
            draw.text((stage_x, processing_y), "⚙️",
                     fill=COLORS["accent2"], anchor="mm")
            draw.text((stage_x, processing_y + 40), "PROCESSING",
                     fill=COLORS["text_white"], anchor="mm")
        elif stage_idx == 1:  # Rendering
            draw.text((stage_x, processing_y), "🎬",
                     fill=COLORS["accent3"], anchor="mm")
            draw.text((stage_x, processing_y + 40), "RENDERING",
                     fill=COLORS["text_white"], anchor="mm")
        elif stage_idx == 2:  # Output
            draw.text((stage_x, processing_y), "✨",
                     fill=COLORS["accent4"], anchor="mm")
            draw.text((stage_x, processing_y + 40), "MASTERING",
                     fill=COLORS["text_white"], anchor="mm")
        
        # Connection arrows
        if stage_idx < 2:
            arrow_start = stage_x + 100
            arrow_end = stage_x + stage_spacing - 100
            arrow_color = [COLORS["accent2"], COLORS["accent3"], COLORS["accent4"]][stage_idx]
            
            draw.line([(arrow_start, processing_y), (arrow_end, processing_y)],
                     fill=arrow_color, width=3)
            # Arrowhead
            draw.polygon([(arrow_end, processing_y),
                        (arrow_end - 15, processing_y - 10),
                        (arrow_end - 15, processing_y + 10)],
                       fill=arrow_color)
    
    # Right reel (output)
    right_reel_x = 1600
    
    draw_film_reel(draw, right_reel_x, reel_y, reel_radius, frame_num * 1.5)
    
    # Output film frames
    for frame_idx in range(3):
        film_x = right_reel_x - reel_radius - 30 - (2 - frame_idx) * (60 + film_spacing)
        
        # Draw film frame
        draw.rectangle(
            [(film_x, film_start_y), (film_x + 60, film_start_y + film_height)],
            fill=COLORS["film"],
            outline=COLORS["accent5"],
            width=2
        )
        
        # Sprocket holes
        for hole_idx in range(4):
            hole_y_pos = film_start_y - 8 - hole_idx * 12
            hole_y_pos_bottom = film_start_y + film_height + 8 + hole_idx * 12
            
            draw.rectangle(
                [(film_x + 5, hole_y_pos), (film_x + 15, hole_y_pos + 5)],
                fill=COLORS["accent5"]
            )
            draw.rectangle(
                [(film_x + 45, hole_y_pos), (film_x + 55, hole_y_pos + 5)],
                fill=COLORS["accent5"]
            )
            draw.rectangle(
                [(film_x + 5, hole_y_pos_bottom), (film_x + 15, hole_y_pos_bottom + 5)],
                fill=COLORS["accent5"]
            )
            draw.rectangle(
                [(film_x + 45, hole_y_pos_bottom), (film_x + 55, hole_y_pos_bottom + 5)],
                fill=COLORS["accent5"]
            )
        
        # Output stage label
        output_idx = 2 + frame_idx
        if output_idx < 5:
            draw.text((film_x + 30, film_start_y + film_height + 25),
                     STAGE_LABELS[output_idx],
                     fill=STAGE_COLORS[output_idx % len(STAGE_COLORS)], anchor="mm")
    
    # Bottom text
    text_y = height - 30
    draw.text((width // 2, text_y),
             "Transform Scripts Into 4K Professional Videos With AI Intelligence",
             fill=COLORS["text_gold"], anchor="mm")
    
    return img

# Generate all frames
print("🎬 Generating cinema film reel animation...")
frames = []

for frame_num in range(FRAMES):
    frame = create_frame(frame_num, FRAMES, WIDTH, HEIGHT)
    frames.append(frame)
    
    if (frame_num + 1) % 30 == 0:
        print(f"  ✓ Frame {frame_num + 1}/{FRAMES}")

# Save as GIF
frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=50,  # 50ms per frame
    loop=0
)

print(f"\n✅ Cinema film reel GIF created: {OUTPUT_PATH}")
print(f"📊 Dimensions: {WIDTH}x{HEIGHT}px")
print(f"🎬 Frames: {len(frames)}")
print(f"⏱️ Duration: {len(frames) * 50 / 1000:.1f} seconds per loop")
print(f"🎨 Features: Rotating reels, film strip, sprocket holes, cinema vintage style")
