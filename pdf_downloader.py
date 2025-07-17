import requests
import tempfile
import os
import logging
from urllib.parse import urlparse
from urllib.request import urlopen
import streamlit as st

class PDFDownloader:
    """Download PDF files from URLs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def download_pdf_from_url(self, url):
        """
        Download a PDF file from a URL.
        
        Args:
            url (str): URL of the PDF file
            
        Returns:
            str: Path to the downloaded temporary file
        """
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError("Invalid URL provided")
            
            # Add progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Downloading PDF from URL...")
            progress_bar.progress(25)
            
            # Download the file
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if content type is PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                # Try to detect PDF by content
                first_bytes = response.content[:8]
                if not first_bytes.startswith(b'%PDF'):
                    raise ValueError("URL does not point to a PDF file")
            
            progress_bar.progress(50)
            status_text.text("Processing downloaded file...")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            progress_bar.progress(75)
            
            # Validate file size
            file_size = os.path.getsize(tmp_file_path)
            if file_size == 0:
                os.unlink(tmp_file_path)
                raise ValueError("Downloaded file is empty")
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                os.unlink(tmp_file_path)
                raise ValueError("File is too large (max 100MB)")
            
            progress_bar.progress(100)
            status_text.text("Download complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return tmp_file_path
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error downloading PDF: {str(e)}")
            raise ValueError(f"Failed to download PDF: Network error")
        except Exception as e:
            self.logger.error(f"Error downloading PDF from URL: {str(e)}")
            raise ValueError(f"Failed to download PDF: {str(e)}")
    
    def _is_valid_url(self, url):
        """Check if the URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def get_file_info_from_url(self, url):
        """
        Get file information from URL without downloading.
        
        Args:
            url (str): URL of the file
            
        Returns:
            dict: File information
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.head(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            content_length = response.headers.get('content-length')
            content_type = response.headers.get('content-type', '')
            
            file_size = int(content_length) if content_length else 0
            
            return {
                'url': url,
                'content_type': content_type,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2) if file_size else 0,
                'is_pdf': 'pdf' in content_type.lower() or url.lower().endswith('.pdf')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file info from URL: {str(e)}")
            return None