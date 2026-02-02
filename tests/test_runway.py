import asyncio
from runwayml import RunwayML

async def test_runway_image():
    client = RunwayML(api_key="key_50d32d6432d622ec0c7c95f1aa0a68cf781192bd531ff1580c3f4853755c5edba0b52fb49426d07aa6b4356e505ab6e1b80987b501aa08f37000fa51f76796b7")
    
    # Try to generate image
    try:
        result = client.text_to_image.create(
            model="gen4_image",
            prompt_text="A beautiful sunset over mountains",
            ratio="1920:1080"
        )
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        print(f"Dir: {[x for x in dir(result) if not x.startswith('_')]}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_runway_image())
