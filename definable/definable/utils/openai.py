import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from definable.media import Audio, File, Image
from definable.utils.log import log_error, log_warning

# Ensure .webp is recognized
mimetypes.add_type("image/webp", ".webp")


def audio_to_message(audio: Sequence[Audio]) -> List[Dict[str, Any]]:
  """
  Add audio to a message for the model. By default, we use the OpenAI audio format but other Models
  can override this method to use a different audio format.

  Automatically normalizes audio formats that OpenAI's ``input_audio`` API
  doesn't accept (e.g. Telegram's ``oga``, generic ``ogg``) to ``wav`` or
  ``mp3`` via :func:`definable.reader.audio.normalize_audio_format`.

  Args:
      audio: Sequence of Audio objects with url, filepath, or content bytes.

  Returns:
      Message content with audio added in the format expected by the model
  """
  from urllib.parse import urlparse

  from definable.reader.audio import normalize_audio_format

  audio_messages = []
  for audio_snippet in audio:
    # Memoized: normalize (ffmpeg) + base64 is the most expensive per-turn media op.
    cached = audio_snippet._block_cache.get("openai_audio")
    if cached is not None:
      audio_messages.append(cached)
      continue
    audio_bytes: Optional[bytes] = None
    audio_format: Optional[str] = audio_snippet.format

    # The audio is raw data
    if audio_snippet.content:
      audio_bytes = audio_snippet.content
      if not audio_format:
        audio_format = "wav"  # Default format if not provided

    # The audio is a URL
    elif audio_snippet.url:
      audio_bytes = audio_snippet.get_content_bytes()
      if audio_bytes is not None and not audio_format:
        # Try to guess format from URL extension
        try:
          parsed_url = urlparse(audio_snippet.url)
          audio_format = Path(parsed_url.path).suffix.lstrip(".")
          if not audio_format:
            log_warning(f"Could not determine audio format from URL path: {parsed_url.path}. Defaulting.")
            audio_format = "wav"
        except Exception as e:
          log_warning(f"Could not determine audio format from URL: {audio_snippet.url}. Error: {e}. Defaulting.")
          audio_format = "wav"

    # The audio is a file path
    elif audio_snippet.filepath:
      path = Path(audio_snippet.filepath)
      if path.exists() and path.is_file():
        try:
          with open(path, "rb") as audio_file:
            audio_bytes = audio_file.read()
          if not audio_format:
            audio_format = path.suffix.lstrip(".")
        except Exception as e:
          log_error(f"Failed to read audio file {path}: {e}")
          continue
      else:
        log_error(f"Audio file not found or is not a file: {path}")
        continue

    # Normalize format and encode for OpenAI input_audio API
    if audio_bytes and audio_format:
      try:
        audio_bytes, audio_format = normalize_audio_format(audio_bytes, audio_format)
      except RuntimeError as e:
        log_error(f"Audio format normalization failed: {e}")
        continue

      encoded_string = base64.b64encode(audio_bytes).decode("utf-8")
      block = {
        "type": "input_audio",
        "input_audio": {
          "data": encoded_string,
          "format": audio_format,
        },
      }
      audio_snippet._block_cache["openai_audio"] = block
      audio_messages.append(block)
    else:
      log_error(f"Could not process audio snippet: {audio_snippet}")

  return audio_messages


def _process_bytes_image(image: bytes, image_format: Optional[str] = None) -> Dict[str, Any]:
  """Process bytes image data."""
  base64_image = base64.b64encode(image).decode("utf-8")

  # Use provided format or attempt detection, defaulting to JPEG
  if image_format:
    mime_type = f"image/{image_format.lower()}"
  else:
    # Try to detect the image format from the bytes
    try:
      import imghdr

      detected_format = imghdr.what(None, h=image)
      mime_type = f"image/{detected_format}" if detected_format else "image/jpeg"
    except Exception:
      mime_type = "image/jpeg"

  image_url = f"data:{mime_type};base64,{base64_image}"
  return {"type": "image_url", "image_url": {"url": image_url}}


def _process_image_path(image_path: Union[Path, str]) -> Dict[str, Any]:
  """Process image ( file path)."""
  # Process local file image
  path = Path(image_path)  # Ensure it's a Path object
  if not path.exists():
    raise FileNotFoundError(f"Image file not found: {image_path}")
  if not path.is_file():
    raise IsADirectoryError(f"Image path is not a file: {image_path}")

  mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"  # Default to jpeg if guess fails
  try:
    with open(path, "rb") as image_file:
      base64_image = base64.b64encode(image_file.read()).decode("utf-8")
      image_url = f"data:{mime_type};base64,{base64_image}"
      return {"type": "image_url", "image_url": {"url": image_url}}
  except Exception as e:
    log_error(f"Failed to read image file {path}: {e}")
    raise  # Re-raise the exception after logging


def _process_image_url(image_url: str) -> Dict[str, Any]:
  """Process image (base64 or URL)."""

  if image_url.startswith("data:image") or image_url.startswith(("http://", "https://")):
    return {"type": "image_url", "image_url": {"url": image_url}}
  else:
    raise ValueError("Image URL must start with 'data:image' or 'http(s)://'.")


def process_image(image: Image) -> Optional[Dict[str, Any]]:
  """Process an image based on the format. Memoized so the base64 encode / disk read
  happens once, not on every agentic turn."""
  cached = image._block_cache.get("openai")
  if cached is not None:
    return cached
  image_payload: Optional[Dict[str, Any]] = None  # Initialize
  try:
    if image.url is not None:
      image_payload = _process_image_url(image.url)

    elif image.filepath is not None:
      image_payload = _process_image_path(image.filepath)

    elif image.content is not None:
      # Pass the format from the Image object
      image_payload = _process_bytes_image(image.content, image.format)

    else:
      log_warning(f"Unsupported image format or no data provided: {image}")
      return None

    if image_payload and image.detail:  # Check if payload was created before adding detail
      # Ensure image_url key exists before trying to access its sub-dictionary
      if "image_url" not in image_payload:
        # Initialize if missing (though unlikely based on helper funcs)
        image_payload["image_url"] = {}
      image_payload["image_url"]["detail"] = image.detail

    if image_payload is not None:
      image._block_cache["openai"] = image_payload
    return image_payload

  except (FileNotFoundError, IsADirectoryError, ValueError) as e:
    log_error(f"Failed to process image due to invalid input: {str(e)}")
    return None  # Return None for handled validation errors
  except Exception as e:
    log_error(f"An unexpected error occurred while processing image: {str(e)}")
    # Depending on policy, you might want to return None or re-raise
    return None  # Return None for unexpected errors as well, preventing crashes


def images_to_message(images: Sequence[Image]) -> List[Dict[str, Any]]:
  """
  Add images to a message for the model. By default, we use the OpenAI image format but other Models
  can override this method to use a different image format.

  Args:
      images: Sequence of images in various formats:
          - str: base64 encoded image, URL, or file path
          - Dict: pre-formatted image data
          - bytes: raw image data

  Returns:
      Message content with images added in the format expected by the model
  """

  # Create a default message content with text
  image_messages: List[Dict[str, Any]] = []

  # Add images to the message content
  for image in images:
    try:
      image_data = process_image(image)
      if image_data:
        image_messages.append(image_data)
    except Exception as e:
      log_error(f"Failed to process image: {str(e)}")
      continue

  return image_messages


def _format_file_for_message(file: File) -> Optional[Dict[str, Any]]:
  """OpenAI file block — memoized so the base64 encode / disk read / URL fetch happens
  once, not on every agentic turn."""
  cached = file._block_cache.get("openai")
  if cached is not None:
    return cached
  block = _format_file_uncached(file)
  if block is not None:
    file._block_cache["openai"] = block
  return block


def _format_file_uncached(file: File) -> Optional[Dict[str, Any]]:
  """
  Add a document url, base64 encoded content or OpenAI file to a message.
  """
  import base64
  import mimetypes
  from pathlib import Path

  # Case 1: Document is a URL
  if file.url is not None:
    from urllib.parse import urlparse

    result = file.file_url_content
    if not result:
      log_error(f"Failed to fetch file from URL: {file.url}")
      return None
    content_bytes, mime_type = result
    name = Path(urlparse(file.url).path).name or "file"
    _mime = mime_type or file.mime_type or mimetypes.guess_type(name)[0] or "application/pdf"
    _encoded = base64.b64encode(content_bytes).decode("utf-8")
    _data_url = f"data:{_mime};base64,{_encoded}"
    return {"type": "file", "file": {"filename": name, "file_data": _data_url}}

  # Case 2: Document is a local file path
  if file.filepath is not None:
    path = Path(file.filepath)
    if not path.is_file():
      log_error(f"File not found: {path}")
      return None
    data = path.read_bytes()

    _mime = file.mime_type or mimetypes.guess_type(path.name)[0] or "application/pdf"
    _encoded = base64.b64encode(data).decode("utf-8")
    _data_url = f"data:{_mime};base64,{_encoded}"
    return {"type": "file", "file": {"filename": path.name, "file_data": _data_url}}

  # Case 3: Document is bytes content
  if file.content is not None:
    name = getattr(file, "filename", "file")
    _mime = file.mime_type or mimetypes.guess_type(name)[0] or "application/pdf"
    _encoded = base64.b64encode(file.content).decode("utf-8")
    _data_url = f"data:{_mime};base64,{_encoded}"
    return {"type": "file", "file": {"filename": name, "file_data": _data_url}}

  return None
