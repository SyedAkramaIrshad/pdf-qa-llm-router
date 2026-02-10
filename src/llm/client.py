"""GLM-4.7 API Client for text and vision calls.

Sources:
- https://docs.bigmodel.cn/cn/guide/models/text/glm-4.7
- https://docs.bigmodel.cn/cn/guide/develop/http/introduction
- https://github.com/zai-org/z-ai-sdk-python
"""

import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import httpx
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..config.settings import get_settings


class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


class GLMClient:
    """Client for Z.ai GLM-4.7 API with text and vision support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        vision_model: Optional[str] = None,
    ):
        """Initialize the GLM client.

        Args:
            api_key: GLM API key (defaults to settings)
            base_url: API base URL (defaults to settings)
            model: Model name for text (defaults to glm-4.7)
            vision_model: Model name for vision (defaults to glm-4v)
        """
        settings = get_settings()

        self.api_key = api_key or settings.glm_api_key
        self.base_url = base_url or settings.glm_base_url
        self.model = model or settings.glm_model
        self.vision_model = vision_model or settings.glm_vision_model

        self.chat_url = f"{self.base_url}chat/completions"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authorization."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API response and check for rate limiting.

        Args:
            response: HTTP response object

        Returns:
            Parsed JSON response

        Raises:
            RateLimitError: If rate limited (429)
            httpx.HTTPStatusError: For other HTTP errors
        """
        if response.status_code == 429:
            raise RateLimitError(f"Rate limit exceeded: {response.text}")
        response.raise_for_status()
        return response.json()

    def _make_request_with_retry(
        self,
        payload: Dict[str, Any],
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic for rate limiting.

        Args:
            payload: Request payload
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (exponential backoff)

        Returns:
            Parsed JSON response
        """
        for attempt in range(max_retries + 1):
            try:
                response = httpx.post(
                    self.chat_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60.0
                )
                return self._handle_response(response)
            except RateLimitError as e:
                if attempt == max_retries:
                    raise
                # Exponential backoff: 1s, 2s, 4s...
                delay = initial_delay * (2 ** attempt)
                print(f"Rate limited. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            except httpx.HTTPStatusError as e:
                if attempt < max_retries and e.response.status_code >= 500:
                    # Retry server errors
                    delay = initial_delay * (2 ** attempt)
                    print(f"Server error {e.response.status_code}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    def _encode_image(self, image_path: str) -> str:
        """Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_pil_image(self, image: Image.Image) -> str:
        """Encode a PIL Image to base64.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: User prompt
            model: Model override
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Enable streaming (not implemented yet)
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }

        result = self._make_request_with_retry(payload)
        return result["choices"][0]["message"]["content"]

    async def generate_text_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Async version of generate_text for parallel calls.

        Args:
            prompt: User prompt
            model: Model override
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        # Async retry logic
        for attempt in range(3 + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.chat_url,
                        headers=self._get_headers(),
                        json=payload,
                    )
                if response.status_code == 429:
                    if attempt == 3:
                        raise RateLimitError(f"Rate limit exceeded: {response.text}")
                    delay = 1.0 * (2 ** attempt)
                    print(f"Rate limited (async). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if attempt < 3 and e.response.status_code >= 500:
                    delay = 1.0 * (2 ** attempt)
                    print(f"Server error {e.response.status_code}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, Path, Image.Image, bytes],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text completion with image input (vision).

        Args:
            prompt: User prompt
            image: Image as file path, PIL Image, or bytes
            model: Model override (defaults to vision model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        # Encode image based on input type
        if isinstance(image, (str, Path)):
            base64_image = self._encode_image(str(image))
        elif isinstance(image, Image.Image):
            base64_image = self._encode_pil_image(image)
        elif isinstance(image, bytes):
            base64_image = base64.b64encode(image).decode("utf-8")
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, PIL.Image, or bytes."
            )

        # Build message with image
        # GLM API expects raw base64, not data URL format
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        payload = {
            "model": model or self.vision_model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        result = self._make_request_with_retry(payload)
        return result["choices"][0]["message"]["content"]

    async def generate_with_image_async(
        self,
        prompt: str,
        image: Union[str, Path, Image.Image, bytes],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Async version of generate_with_image.

        Args:
            prompt: User prompt
            image: Image as file path, PIL Image, or bytes
            model: Model override (defaults to vision model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        # Encode image based on input type
        if isinstance(image, (str, Path)):
            base64_image = self._encode_image(str(image))
        elif isinstance(image, Image.Image):
            base64_image = self._encode_pil_image(image)
        elif isinstance(image, bytes):
            base64_image = base64.b64encode(image).decode("utf-8")
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, PIL.Image, or bytes."
            )

        # Build message with image
        # GLM API expects raw base64, not data URL format
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        payload = {
            "model": model or self.vision_model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Async retry logic
        for attempt in range(3 + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.chat_url,
                        headers=self._get_headers(),
                        json=payload,
                    )
                if response.status_code == 429:
                    if attempt == 3:
                        raise RateLimitError(f"Rate limit exceeded: {response.text}")
                    delay = 1.0 * (2 ** attempt)
                    print(f"Rate limited (async). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if attempt < 3 and e.response.status_code >= 500:
                    delay = 1.0 * (2 ** attempt)
                    print(f"Server error {e.response.status_code}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON output with improved error handling.

        Args:
            prompt: User prompt (should request JSON output)
            model: Model override
            temperature: Sampling temperature (lower for more deterministic)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON dictionary
        """
        from .schemas import safe_parse_json

        # Add JSON format instruction if not present
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nRespond ONLY with valid JSON. No markdown, no explanation."

        try:
            response_text = self.generate_text(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "You always respond with valid JSON only.",
            )

            return safe_parse_json(response_text)

        except Exception as e:
            print(f"  Error generating JSON: {e}")
            return {
                "summary": [f"Error: {str(e)}"],
                "keywords": [],
                "insights": []
            }

    async def generate_json_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """Async version of generate_json with improved error handling.

        Args:
            prompt: User prompt (should request JSON output)
            model: Model override
            temperature: Sampling temperature (lower for more deterministic)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            retries: Number of retries on parse failure

        Returns:
            Parsed JSON dictionary
        """
        from .schemas import safe_parse_json

        # Add JSON format instruction if not present
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nRespond ONLY with valid JSON. No markdown, no explanation."

        for attempt in range(retries + 1):
            try:
                response_text = await self.generate_text_async(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt or "You always respond with valid JSON only.",
                )

                result = safe_parse_json(response_text)

                # Check if we got a valid response (not just error fallback)
                if "Failed to parse" in result.get("summary", [""])[0]:
                    if attempt < retries:
                        print(f"  Retrying JSON parse (attempt {attempt + 2}/{retries + 1})...")
                        continue

                return result

            except Exception as e:
                if attempt < retries:
                    print(f"  Error generating JSON: {e}. Retrying...")
                    await asyncio.sleep(1)
                else:
                    print(f"  Failed after {retries + 1} attempts: {e}")
                    return {
                        "summary": [f"Error: {str(e)}"],
                        "keywords": [],
                        "insights": []
                    }

        return {"summary": ["Unknown error"], "keywords": [], "insights": []}


# Convenience function to get a client instance
def get_client() -> GLMClient:
    """Get a configured GLM client instance."""
    return GLMClient()
