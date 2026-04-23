from __future__ import annotations

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class ImageGenerationTool:
    name = "generate_image"
    description = "Generate an image from a prompt"

    def __init__(self):
        self.client = OpenAI() if OpenAI else None

    async def run(self, prompt: str, size="1024x1024", quality="standard"):
        if self.client is None:
            return "OpenAI not installed"

        try:
            result = self.client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                quality=quality,
            )
            return result.data[0].url
        except Exception as e:
            return str(e)
