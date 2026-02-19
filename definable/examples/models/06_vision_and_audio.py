"""
Multimodal inputs: Vision and Audio.

This example shows how to:
- Send images to vision-capable models
- Send audio to audio-capable models
- Work with different media types

Requirements:
    export OPENAI_API_KEY=sk-...

Note: This example requires vision/audio capable models like gpt-4o or gpt-4o-mini.
"""

from pathlib import Path

from definable.media import Audio, Image
from definable.model.message import Message
from definable.model.openai import OpenAIChat


def image_from_url():
  """Analyze an image from a URL."""
  model = OpenAIChat(id="gpt-4o-mini")

  print("Image Analysis (URL)")
  print("-" * 40)

  # Create an image from URL
  image = Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg")

  # Send message with image
  response = model.invoke(
    messages=[
      Message(
        role="user",
        content="What do you see in this image? Describe it briefly.",
        images=[image],
      ),
    ],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Response: {response.content}")
  print()


def image_from_file():
  """Analyze an image from a local file."""
  model = OpenAIChat(id="gpt-4o-mini")

  print("Image Analysis (Local File)")
  print("-" * 40)

  # Example with a local file path
  # Replace with an actual image path on your system
  image_path = Path("./example_image.png")

  if not image_path.exists():
    print(f"Note: Create an image at {image_path} to test local file analysis.")
    print("Skipping local file example...")
    print()
    return

  # Create image from file path
  image = Image(filepath=str(image_path))

  response = model.invoke(
    messages=[
      Message(
        role="user",
        content="Describe this image in detail.",
        images=[image],
      ),
    ],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Response: {response.content}")
  print()


def image_from_base64():
  """Analyze an image from base64 data."""
  model = OpenAIChat(id="gpt-4o-mini")

  print("Image Analysis (Base64)")
  print("-" * 40)

  # Example base64 image (1x1 red pixel PNG for demonstration)
  # In practice, you'd have actual image data
  base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

  # Use from_base64 classmethod to create image from base64 string
  image = Image.from_base64(
    base64_data,
    mime_type="image/png",
  )

  response = model.invoke(
    messages=[
      Message(
        role="user",
        content="What color is this image?",
        images=[image],
      ),
    ],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Response: {response.content}")
  print()


def multiple_images():
  """Analyze multiple images in one request."""
  model = OpenAIChat(id="gpt-4o-mini")

  print("Multiple Images Analysis")
  print("-" * 40)

  # Create multiple images
  images = [
    Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"),
    Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"),
  ]

  response = model.invoke(
    messages=[
      Message(
        role="user",
        content="Compare these two images. What animals do you see?",
        images=images,
      ),
    ],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Response: {response.content}")
  print()


def audio_input():
  """Process audio input (requires audio-capable model)."""
  model = OpenAIChat(id="gpt-4o-audio-preview")  # Audio-capable model

  print("Audio Input")
  print("-" * 40)

  # Example with audio file
  audio_path = Path("./example_audio.mp3")

  if not audio_path.exists():
    print(f"Note: Create an audio file at {audio_path} to test audio input.")
    print("Skipping audio example...")
    print()
    return

  # Create audio from file
  audio = Audio(filepath=str(audio_path))

  response = model.invoke(
    messages=[
      Message(
        role="user",
        content="What is being said in this audio?",
        audio=[audio],
      ),
    ],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Response: {response.content}")
  print()


def image_with_detail_control():
  """Control image detail level for vision analysis."""
  model = OpenAIChat(id="gpt-4o-mini")

  print("Image with Detail Control")
  print("-" * 40)

  # Use 'low' detail for faster, cheaper processing
  # Use 'high' detail for more detailed analysis
  image = Image(
    url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    detail="low",  # or "high" or "auto"
  )

  response = model.invoke(
    messages=[
      Message(
        role="user",
        content="What is in this image?",
        images=[image],
      ),
    ],
    assistant_message=Message(role="assistant", content=""),
  )

  print(f"Response (low detail): {response.content}")
  print()


if __name__ == "__main__":
  image_from_url()
  image_from_file()
  image_from_base64()
  multiple_images()
  audio_input()
  image_with_detail_control()
