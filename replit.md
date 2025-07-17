# PDF Figure Extraction & Classification Tool

## Overview

This is a Streamlit-based web application that automatically extracts and classifies figures from PDF documents. The tool uses PyMuPDF for PDF processing, PIL for image manipulation, and implements a custom figure classification system. Users can upload PDF files, extract all embedded figures, and get them classified into different categories with an intuitive web interface.

## User Preferences

Preferred communication style: Simple, everyday language.
Cost preference: Free solutions preferred over paid APIs.
Feature request: URL-based PDF processing in addition to file upload.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Architecture Pattern**: Single-page application with session state management
- **UI Components**: File uploader, sidebar navigation, main content area for results display
- **State Management**: Streamlit session state for maintaining extracted figures and classification results

### Backend Architecture
- **Processing Pipeline**: Modular design with separate components for extraction and classification
- **Core Components**:
  - `PDFFigureExtractor`: Handles PDF parsing and figure extraction using PyMuPDF
  - `FigureClassifier`: Implements rule-based and feature-based figure classification
  - `utils`: Provides utility functions for file handling and image processing

### Data Processing Flow
1. PDF upload through Streamlit file uploader
2. PDF processing using PyMuPDF (fitz) to extract embedded images
3. Image filtering (minimum size requirements, format validation)
4. Feature extraction from images for classification
5. Rule-based classification into predefined categories
6. Results display with download capabilities

## Key Components

### PDF Processing (`figure_extractor.py`)
- **Purpose**: Extract figures and images from PDF documents
- **Technology**: PyMuPDF (fitz library)
- **Key Features**:
  - Page-by-page image extraction
  - Vector graphics extraction
  - Image size filtering (minimum 50x50 pixels)
  - Format conversion (CMYK to RGB)
  - PIL Image integration

### PDF Download (`pdf_downloader.py`)
- **Purpose**: Download PDF files from URLs
- **Technology**: Requests library with proper headers
- **Key Features**:
  - URL validation and file type checking
  - Progress indication during download
  - File size limits (100MB max)
  - Proper cleanup of temporary files

### AI Figure Classification (`ai_classifier.py`)
- **Purpose**: Classify extracted figures into categories using AI
- **Approach**: Google Gemini AI-powered visual classification
- **Technology**: Google Gemini 2.0 Flash (Free tier with rate limiting protection)
- **Features**:
  - Comprehensive figure type detection (20+ categories)
  - High accuracy AI-powered classification
  - Detailed descriptions and reasoning
  - Confidence scoring
  - Support for charts, diagrams, images, tables, maps, etc.
  - Exponential backoff retry logic for rate limiting
  - Intelligent fallback for quota issues

### Web Interface (`app.py`)
- **Framework**: Streamlit
- **Layout**: Wide layout with configuration sidebar and dual input methods
- **Input Methods**: File upload + URL download support
- **API Key Management**: User-provided API keys for personalized rate limits
- **Session Management**: Persistent state for extracted figures and results
- **User Experience**: Progress indication, file size display, AI classification feedback
- **Features**: PDF URL validation, enhanced figure cards with AI descriptions, self-service API key input

### PDF Report Generation (`report_generator.py`)
- **Purpose**: Generate comprehensive PDF analysis reports
- **Technology**: ReportLab for PDF creation with professional layouts
- **Key Features**:
  - Executive summary with key statistics
  - Figure type distribution tables
  - Confidence score analysis
  - Individual figure thumbnails with descriptions
  - Professional formatting with charts and tables

### Utilities (`utils.py`)
- **File Operations**: File size calculation, download link generation
- **Image Processing**: Resize for display, validation, metadata extraction
- **Display Functions**: Enhanced emoji mapping and figure type formatting for 20+ categories
- **Helper Functions**: Support for main application workflow

## Data Flow

1. **Input**: User uploads PDF file or provides URL through Streamlit interface
2. **Download**: If URL provided, PDF downloaded via PDFDownloader
3. **Extraction**: PyMuPDF processes PDF, extracts embedded images
4. **Filtering**: Images filtered by size and format requirements
5. **AI Classification**: Each image analyzed using Google Gemini AI
6. **Storage**: Results stored in Streamlit session state with detailed metadata
7. **Display**: Processed figures displayed with AI classifications, descriptions, and reasoning
8. **Export**: Users can download individual figures or batch results

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **PyMuPDF (fitz)**: PDF processing and image extraction
- **PIL (Pillow)**: Image processing and manipulation
- **pandas**: Data structure management
- **google-genai**: Google Gemini AI integration
- **requests**: HTTP client for URL downloads
- **urllib3**: URL handling utilities
- **reportlab**: PDF report generation
- **numpy**: Numerical computing

### System Dependencies
- **tempfile**: Temporary file handling
- **zipfile**: Archive creation for bulk downloads
- **io**: In-memory file operations
- **os**: Operating system interface
- **logging**: Application logging

## Deployment Strategy

### Development Environment
- **Platform**: Designed for Replit deployment
- **Configuration**: Streamlit configuration for web deployment
- **File Structure**: Modular Python application with clear separation of concerns

### Production Considerations
- **Scalability**: Session-based processing suitable for single-user instances
- **Memory Management**: Temporary file handling for large PDF processing
- **Error Handling**: Comprehensive logging and validation throughout pipeline
- **Performance**: Image size filtering and efficient PDF processing

### Deployment Requirements
- Python 3.7+ environment
- Sufficient memory for PDF and image processing
- Web server capability (provided by Streamlit)
- File system access for temporary file operations

## Technical Architecture Decisions

### PDF Processing Library Choice
- **Decision**: PyMuPDF (fitz) for PDF processing
- **Rationale**: Robust image extraction capabilities, good performance, PIL integration
- **Alternatives**: PDFPlumber, PyPDF2 (limited image extraction capabilities)

### Classification Approach
- **Decision**: AI-powered classification using Google Gemini with improved error handling
- **Rationale**: High accuracy, comprehensive figure type detection, detailed descriptions
- **Technology**: Google Gemini 2.0 Flash with exponential backoff retry logic
- **Benefits**: 20+ figure categories, natural language descriptions, reasoning explanation, graceful rate limit handling

### Web Framework Selection
- **Decision**: Streamlit for rapid prototyping and deployment
- **Rationale**: Quick development, built-in state management, suitable for data applications
- **Trade-offs**: Limited customization compared to Flask/Django but faster development

### Image Processing Strategy
- **Decision**: PIL for image manipulation with AI for classification
- **Rationale**: PIL for basic operations, Google Gemini AI for intelligent classification
- **Benefits**: High accuracy classification, natural language descriptions, minimal local processing

## Recent Changes (2025-07-17)

### Major Architecture Updates
- **AI Classification System**: Evolved from rule-based to Gemini with improved error handling
  - Implemented comprehensive 20+ figure type detection
  - Added detailed descriptions and reasoning for each classification
  - Integrated confidence scoring with visual indicators
  - Added exponential backoff retry logic for rate limiting
  - Improved fallback mechanisms for quota issues
  
- **URL Upload Feature**: Added PDF processing from URLs
  - Implemented PDFDownloader class for secure URL downloads
  - Added URL validation and file type checking
  - Created tabbed interface for file upload vs URL input
  
- **Enhanced User Interface**: 
  - Updated figure cards with AI descriptions and reasoning
  - Added expandable sections for classification details
  - Improved visual indicators for confidence levels
  - Enhanced welcome screen with dual input method instructions

- **PDF Report Generation**: Added comprehensive analysis report feature
  - Professional PDF reports with executive summary
  - Statistical analysis and confidence distribution
  - Figure type distribution with examples
  - Individual figure thumbnails with detailed descriptions
  - Downloadable reports with timestamps

### Technical Improvements
- **Enhanced Solution**: Uses Google Gemini with intelligent rate limiting and retry logic
- **Error Handling**: Comprehensive error handling for URL downloads and AI classification
- **User Experience**: Progress indicators for both file upload and URL processing
- **Memory Management**: Proper cleanup of temporary files for URL downloads
- **Professional Reports**: Comprehensive PDF reports with executive summaries and detailed analysis
- **User API Key Support**: Users can now provide their own Gemini API keys for better rate limits and personalized service