"""PDF Processor for extracting text and images in chunks.

This module handles:
- PDF text extraction with pdfplumber
- Image extraction with pdf2image
- 10-page chunking strategy
- Metadata generation
"""

import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib

import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
from pypdf import PdfReader

from ..config.settings import get_settings


class PDFProcessor:
    """Process PDFs into text and image chunks."""

    def __init__(self, chunk_size: Optional[int] = None):
        """Initialize the PDF processor.

        Args:
            chunk_size: Number of pages per chunk (default from settings)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.settings = settings

    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract basic PDF metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF metadata
        """
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        # Calculate total sections
        total_sections = (total_pages + self.chunk_size - 1) // self.chunk_size

        metadata = {
            "total_pages": total_pages,
            "total_sections": total_sections,
            "chunk_size": self.chunk_size,
            "filename": Path(pdf_path).name,
            "file_hash": self._get_file_hash(pdf_path),
        }

        # Add PDF metadata if available
        if reader.metadata:
            metadata["title"] = reader.metadata.get("/Title", "")
            metadata["author"] = reader.metadata.get("/Author", "")
            metadata["creator"] = reader.metadata.get("/Creator", "")

        return metadata

    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for caching."""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def extract_section_text(
        self,
        pdf_path: str,
        section_id: int
    ) -> Dict[str, Any]:
        """Extract text from a specific section (chunk).

        Args:
            pdf_path: Path to PDF file
            section_id: Section number (0-indexed)

        Returns:
            Dictionary with section data
        """
        metadata = self.get_pdf_metadata(pdf_path)

        if section_id >= metadata["total_sections"]:
            raise ValueError(
                f"Section {section_id} out of range. "
                f"PDF has {metadata['total_sections']} sections."
            )

        # Calculate page range
        start_page = section_id * self.chunk_size
        end_page = min((section_id + 1) * self.chunk_size, metadata["total_pages"])

        # Extract text from pages
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            full_text = []

            for page_num in range(start_page, end_page):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""

                pages_text.append({
                    "page_number": page_num + 1,  # 1-indexed for user
                    "text": text,
                    "char_count": len(text)
                })
                full_text.append(text)

        return {
            "section_id": section_id + 1,  # 1-indexed
            "page_range": [start_page + 1, end_page],  # 1-indexed
            "pages": pages_text,
            "full_text": "\n\n".join(full_text),
            "total_chars": sum(len(t) for t in full_text),
        }

    def extract_page_images(
        self,
        pdf_path: str,
        page_number: int,
        dpi: int = 150
    ) -> List[Dict[str, Any]]:
        """Extract images from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            dpi: Resolution for image conversion

        Returns:
            List of image dictionaries with PIL Image and base64
        """
        # Convert page to image
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi
        )

        result = []
        for i, img in enumerate(images):
            # Convert to base64 for API
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            result.append({
                "image": img,  # PIL Image for saving/display
                "base64": base64_data,  # For API calls
                "size": img.size,
                "mode": img.mode,
            })

        return result

    def extract_section_images(
        self,
        pdf_path: str,
        section_id: int,
        dpi: int = 150
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Extract images from all pages in a section.

        Args:
            pdf_path: Path to PDF file
            section_id: Section number (0-indexed)
            dpi: Resolution for image conversion

        Returns:
            Dictionary mapping page_number (1-indexed) to list of images
        """
        metadata = self.get_pdf_metadata(pdf_path)

        # Calculate page range
        start_page = section_id * self.chunk_size
        end_page = min((section_id + 1) * self.chunk_size, metadata["total_pages"])

        result = {}

        for page_num in range(start_page, end_page):
            user_page_num = page_num + 1  # 1-indexed
            try:
                images = self.extract_page_images(pdf_path, user_page_num, dpi)
                if images:
                    result[user_page_num] = images
            except Exception as e:
                # If image extraction fails, continue with text-only
                print(f"Warning: Failed to extract images from page {user_page_num}: {e}")
                result[user_page_num] = []

        return result

    def extract_page_content(
        self,
        pdf_path: str,
        page_number: int,
        include_images: bool = True,
        dpi: int = 150
    ) -> Dict[str, Any]:
        """Extract all content (text + images) from a single page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            include_images: Whether to extract images
            dpi: Resolution for image conversion

        Returns:
            Dictionary with page content
        """
        metadata = self.get_pdf_metadata(pdf_path)

        if page_number < 1 or page_number > metadata["total_pages"]:
            raise ValueError(
                f"Page {page_number} out of range. "
                f"PDF has {metadata['total_pages']} pages."
            )

        # Extract text
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]  # Convert to 0-indexed
            text = page.extract_text() or ""

            # Also check for images within the page
            page_images = page.images
            inline_image_count = len(page_images)

        result = {
            "page_number": page_number,
            "text": text,
            "char_count": len(text),
            "inline_image_count": inline_image_count,
        }

        # Extract full page images if requested
        if include_images:
            try:
                images = self.extract_page_images(pdf_path, page_number, dpi)
                result["images"] = images
                result["has_images"] = len(images) > 0
            except Exception as e:
                result["images"] = []
                result["has_images"] = False
                result["image_error"] = str(e)

        return result

    def get_all_sections(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Get list of all sections with their page ranges.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of section dictionaries
        """
        metadata = self.get_pdf_metadata(pdf_path)
        sections = []

        for i in range(metadata["total_sections"]):
            start_page = i * self.chunk_size + 1  # 1-indexed
            end_page = min((i + 1) * self.chunk_size, metadata["total_pages"])

            sections.append({
                "section_id": i + 1,  # 1-indexed
                "page_range": [start_page, end_page],
                "page_count": end_page - start_page + 1,
            })

        return sections


def get_processor() -> PDFProcessor:
    """Get a configured PDF processor instance."""
    return PDFProcessor()
