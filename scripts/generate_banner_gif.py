#!/usr/bin/env python
"""Generate animated GIF banner for AIPROD README"""

from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
WIDTH = 1920
HEIGHT = 340
FRAMES = 60
OUTPUT_PATH = "assets/banner-animation.gif"

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Colors
BG_COLOR = "#0a0a0a"
COLORS = ["#FF006E", "#8338EC", "#3A86FF", "#FFBE0B", "#FB5607"]
ACCENT_COLORS = ["#FF006E", "#FB5607", "#FFBE0B", "#3A86FF", "#8338EC"]
BORDER_COLOR = "#4A90E2"
TEXT_COLOR = "#FFFFFF"

frames = []

for frame_num in range(FRAMES):
    # Create image with gradient-like dark background
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Draw border
    border_width = 3
    draw.rectangle(
        [(border_width, border_width), (WIDTH - border_width, HEIGHT - border_width)],
        outline=BORDER_COLOR,
        width=border_width
    )
    
    # Calculate animation progress (0 to 1)
    progress = (frame_num % FRAMES) / FRAMES
    
    # Draw title
    title_y = 30
    draw.text((WIDTH // 2, title_y), "🎬 AIPROD - VIDEO AI PIPELINE", 
              fill=TEXT_COLOR, anchor="mm")
    
    # Draw subtitle
    subtitle_y = 80
    draw.text((WIDTH // 2, subtitle_y), 
              "Transform Scripts Into Professional 4K Videos with AI Intelligence",
              fill="#FFBE0B", anchor="mm")
    
    # Stage positions
    stages = [
        (300, "📝\nSCRIPTS"),
        (700, "🧠\nINTELLIGENCE"),
        (1100, "🎬\nCREATION"),
        (1500, "✨\nEXCELLENCE"),
    ]
    
    # Draw pipeline stages
    stage_y = 180
    radius = 50
    
    for idx, (x, label) in enumerate(stages):
        # Calculate if this stage is active based on animation progress
        stage_progress = (progress * len(stages) - idx)
        
        if stage_progress >= 0:
            # Draw circle
            intensity = min(1.0, max(0, stage_progress))
            alpha = int(255 * intensity)
            
            # Draw filled circle
            color = COLORS[idx]
            draw.ellipse(
                [(x - radius, stage_y - radius), (x + radius, stage_y + radius)],
                fill=color
            )
            
            # Draw label
            draw.text((x, stage_y), label.split("\n")[0], 
                     fill=TEXT_COLOR, anchor="mm")
            try:
                draw.text((x, stage_y + 35), label.split("\n")[1], 
                         fill=TEXT_COLOR, anchor="mm")
            except:
                pass
            
            # Draw arrow to next stage if not last
            if idx < len(stages) - 1:
                next_x = stages[idx + 1][0]
                arrow_x_start = x + radius + 20
                arrow_x_end = next_x - radius - 20
                arrow_y = stage_y
                
                # Draw arrow line
                draw.line([(arrow_x_start, arrow_y), (arrow_x_end, arrow_y)],
                         fill=ACCENT_COLORS[idx], width=4)
                
                # Draw arrow head
                arrow_size = 15
                draw.polygon([(arrow_x_end, arrow_y),
                            (arrow_x_end - arrow_size, arrow_y - arrow_size),
                            (arrow_x_end - arrow_size, arrow_y + arrow_size)],
                           fill=ACCENT_COLORS[idx])
    
    # Draw final stage (rocket)
    final_x = 1860
    final_progress = (progress * len(stages) - (len(stages) - 1))
    if final_progress >= 0:
        draw.ellipse(
            [(final_x - radius, stage_y - radius), (final_x + radius, stage_y + radius)],
            fill=COLORS[-1]
        )
        draw.text((final_x, stage_y), "🚀\nLAUNCH",
                 fill=TEXT_COLOR, anchor="mm")
    
    # Draw progress bar at bottom
    progress_bar_y = HEIGHT - 40
    progress_bar_height = 6
    progress_bar_width = WIDTH - 60
    
    # Background bar
    draw.rectangle(
        [(30, progress_bar_y), (30 + progress_bar_width, progress_bar_y + progress_bar_height)],
        outline=BORDER_COLOR,
        width=2
    )
    
    # Progress fill
    fill_width = progress_bar_width * progress
    draw.rectangle(
        [(30, progress_bar_y), (30 + fill_width, progress_bar_y + progress_bar_height)],
        fill="#4A90E2"
    )
    
    # Add percentage text
    percentage = int(progress * 100)
    draw.text((WIDTH // 2, progress_bar_y + progress_bar_height + 15),
             f"Pipeline Progress: {percentage}%",
             fill=TEXT_COLOR, anchor="mm")
    
    frames.append(img)

# Save as GIF
frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=50,  # 50ms per frame
    loop=0  # Loop forever
)

print(f"✅ GIF banner created: {OUTPUT_PATH}")
print(f"📊 Dimensions: {WIDTH}x{HEIGHT}px")
print(f"🎬 Frames: {len(frames)}")
print(f"⏱️ Duration: {len(frames) * 50 / 1000:.1f} seconds per loop")
