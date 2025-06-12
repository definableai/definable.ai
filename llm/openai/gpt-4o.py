import asyncio
import base64
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.responses import Response


class GPT4o:
  def __init__(self, api_key: str | None = None):
    self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    self.client = AsyncOpenAI(api_key=self.api_key)

  async def ainvoke(self, prompt: str) -> str:
    response = await self.client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": prompt}],
      stream=True,
    )
    async for chunk in response:
      print(chunk)
      print(chunk.choices[0].delta.content, end="", flush=True)
    return ""

  async def agenerate_image(self, prompt: str, output_path: Optional[str] = None, model: str = "gpt-image-1") -> Optional[str]:
    """
    Generate an image using OpenAI's image generation capabilities.

    Args:
        prompt: The text description of the image to generate
        output_path: Optional path to save the generated image. If None, returns base64 string
        model: The model to use for image generation

    Returns:
        Optional[str]: Base64 encoded image data if output_path is None, None otherwise
    """
    try:
      # The actual API call
      response = await self.client.images.generate(model=model, prompt=prompt, n=1, size="1024x1024")
      print(response)
      # Extract the base64 data
      if not response.data or len(response.data) == 0:
        return None

      # Get the base64 data
      image_base64 = response.data[0].b64_json
      if not image_base64:
        return None

      if output_path:
        # Save base64 data to file if output_path is provided
        image_data = base64.b64decode(image_base64)
        with open(output_path, "wb") as f:
          f.write(image_data)
        return None

      return image_base64

    except Exception as e:
      print(f"Error generating image: {e}")
      return None


if __name__ == "__main__":

  async def main():
    gpt4o = GPT4o()
    # Example usage
    image_data = await gpt4o.agenerate_image(
      "Generate photorealistic image of a AI researcher explaining something on the board about AI and the board is on the wall and reflection of tajmahal is on the board",
      output_path="testing.png",
    )
    if image_data:
      print("Image generated successfully!")
    else:
      print("Failed to generate image")

  asyncio.run(main())
