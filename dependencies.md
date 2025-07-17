# PDF Figure Extraction & Classification Tool Dependencies

## Current Installed Packages

The following packages are currently installed and managed by Replit:

### Core Dependencies
- **streamlit** - Web application framework for the user interface
- **PyMuPDF (fitz)** - PDF processing and figure extraction
- **Pillow** - Image processing and manipulation
- **opencv-python** - Computer vision and image analysis
- **google-genai** - Google Gemini AI integration for figure classification
- **numpy** - Numerical computing for image analysis
- **pandas** - Data structures and analysis
- **requests** - HTTP client for PDF downloads from URLs
- **urllib3** - URL handling utilities
- **reportlab** - PDF report generation
- **scikit-learn** - Machine learning utilities
- **trafilatura** - Web content extraction
- **joblib** - Parallel processing utilities

### Python Standard Library Modules Used
- os, io, json, logging, tempfile, zipfile, datetime, uuid, time, random, base64

## Installation Notes

Replit automatically manages package installation through the pyproject.toml file. Dependencies are installed using the package manager tool rather than manually editing requirements.txt.

## Optional Dependencies

- **openai** - OpenAI API integration (currently not used but available for future enhancements)

## System Requirements

- Python 3.11+
- Sufficient memory for PDF processing (handled by Replit environment)
- Network access for API calls and URL downloads