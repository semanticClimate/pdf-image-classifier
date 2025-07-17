import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import os
import tempfile
import zipfile
import io
from datetime import datetime
from PIL import Image
import pandas as pd
from figure_extractor import PDFFigureExtractor
from ai_classifier import AIFigureClassifier
from pdf_downloader import PDFDownloader
from report_generator import PDFReportGenerator
from utils import create_download_link, get_file_size, format_figure_type, get_figure_type_emoji

# Initialize session state
if 'extracted_figures' not in st.session_state:
    st.session_state.extracted_figures = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'source_info' not in st.session_state:
    st.session_state.source_info = "PDF Document"
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = None


def main():
    st.set_page_config(
        page_title="PDF Figure Extraction & Classification Tool",
        page_icon="📊",
        layout="wide")

    st.title("📊 FigSense: PDF Figure Extraction & Classification Tool")
    st.markdown(
        "Upload a PDF document to automatically extract and classify all figures within it."
    )

    # Sidebar for file upload and API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input section with improved validation
        with st.expander("🔑 Gemini API Key", expanded=True):
            st.markdown("""
            **For best results, provide your Gemini API key:**
            1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Create a new API key
            3. Paste it below
            """)
            
            user_api_key = st.text_input(
                "API Key", 
                type="password",
                placeholder="Enter your Gemini API key...",
                help="Get your API key from https://aistudio.google.com/app/apikey"
            )
            
            # Test API key button
            if user_api_key:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Test API Key", type="secondary"):
                        test_api_key(user_api_key)
                
                with col2:
                    if st.button("💾 Save Key", type="primary"):
                        st.session_state.user_api_key = user_api_key
                        st.session_state.api_key_status = "saved"
                        st.success("✅ API key saved!")
                        st.rerun()
            
            # Display API key status
            if hasattr(st.session_state, 'user_api_key') and st.session_state.user_api_key:
                if st.session_state.api_key_status == "valid":
                    st.success("✅ API key is valid and ready to use!")
                elif st.session_state.api_key_status == "invalid":
                    st.error("❌ API key is invalid. Please check and try again.")
                elif st.session_state.api_key_status == "saved":
                    st.info("💾 API key saved. Click 'Test API Key' to validate.")
                else:
                    st.info("🔑 API key provided. Click 'Test API Key' to validate.")
            else:
                st.warning("⚠️ No API key provided. Will use basic heuristic classification.")
        
        st.header("📁 PDF Input Options")

        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📁 Upload File", "🔗 From URL"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload a PDF document to extract and classify figures")

            if uploaded_file is not None:
                file_size = get_file_size(uploaded_file)
                st.info(f"File size: {file_size}")

                if st.button("Process Uploaded PDF", type="primary"):
                    process_pdf(uploaded_file)

        with tab2:
            pdf_url = st.text_input(
                "Enter PDF URL",
                placeholder="https://example.com/document.pdf",
                help="Enter the direct URL to a PDF file")

            if pdf_url:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Validate URL", type="secondary"):
                        validate_pdf_url(pdf_url)
                
                with col2:
                    if st.button("🚀 Process PDF", type="primary"):
                        process_pdf_from_url(pdf_url)

    # Main content area
    if st.session_state.processing_complete and st.session_state.extracted_figures:
        display_results()
    else:
        display_welcome_screen()


def test_api_key(api_key):
    """Test the provided API key."""
    try:
        with st.spinner("Testing API key..."):
            # Test by creating a classifier instance
            classifier = AIFigureClassifier(api_key=api_key)
            
            if classifier.is_api_available():
                st.session_state.api_key_status = "valid"
                st.success("✅ API key is valid and working!")
            else:
                st.session_state.api_key_status = "invalid"
                st.error("❌ API key validation failed.")
                
    except ValueError as e:
        st.session_state.api_key_status = "invalid"
        st.error(f"❌ API key error: {str(e)}")
    except Exception as e:
        st.session_state.api_key_status = "invalid"
        st.error(f"❌ Unexpected error: {str(e)}")


def process_pdf(uploaded_file, from_url=False, url=None):
    """Process the uploaded PDF file and extract/classify figures."""
    
    # Check if we have a valid API key
    api_key = st.session_state.get('user_api_key')
    if not api_key:
        st.info("ℹ️ Processing without API key - using heuristic classification")
    elif st.session_state.get('api_key_status') != 'valid':
        st.warning("⚠️ API key not validated - results may vary")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Initialize components
        extractor = PDFFigureExtractor()
        
        # Initialize AI classifier with detailed error handling
        try:
            classifier = AIFigureClassifier(api_key=api_key)
            
            # Show classification method being used
            if classifier.is_api_available():
                st.success("🤖 Using Gemini AI for advanced classification")
            else:
                st.info("🔧 Using heuristic classification (faster but less accurate)")
                
        except ValueError as e:
            st.error(f"❌ Classifier initialization failed: {str(e)}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error initializing classifier: {str(e)}")
            return

        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Extract figures
        status_text.text("📄 Extracting figures from PDF...")
        progress_bar.progress(25)

        try:
            extracted_figures = extractor.extract_figures(tmp_file_path)
        except Exception as e:
            st.error(f"❌ Error extracting figures: {str(e)}")
            os.unlink(tmp_file_path)
            return

        if not extracted_figures:
            st.warning("⚠️ No figures found in the PDF document.")
            os.unlink(tmp_file_path)
            return

        progress_bar.progress(50)
        status_text.text(f"🔍 Found {len(extracted_figures)} figures. Starting classification...")

        # Classify figures using AI with enhanced progress tracking
        classification_results = []
        
        for i, figure_data in enumerate(extracted_figures):
            # Update progress for each figure
            current_progress = 50 + (i * 40) // len(extracted_figures)
            progress_bar.progress(current_progress)
            
            status_text.text(f"🔍 Classifying figure {i + 1}/{len(extracted_figures)}...")
            
            try:
                classification_result = classifier.classify_figure(figure_data['image'])
                
                # Add metadata
                classification_result.update({
                    'figure_id': i,
                    'page': figure_data['page'],
                    'bbox': figure_data['bbox']
                })
                
                classification_results.append(classification_result)
                
            except Exception as e:
                st.warning(f"⚠️ Error classifying figure {i + 1}: {str(e)}")
                # Add fallback result
                classification_results.append({
                    'figure_id': i,
                    'classification': 'unknown',
                    'confidence': 0.1,
                    'description': f'Classification failed: {str(e)}',
                    'details': {},
                    'reasoning': 'Error during classification',
                    'method': 'error',
                    'page': figure_data['page'],
                    'bbox': figure_data['bbox']
                })

        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")

        # Store results in session state
        st.session_state.extracted_figures = extracted_figures
        st.session_state.classification_results = classification_results
        st.session_state.processing_complete = True

        # Store source information
        if from_url:
            st.session_state.source_info = f"PDF from URL: {url}"
        else:
            st.session_state.source_info = f"Uploaded PDF: {uploaded_file.name if hasattr(uploaded_file, 'name') else 'Unknown'}"

        # Clean up temporary file
        os.unlink(tmp_file_path)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show completion message
        source_info = f"from URL: {url}" if from_url else "from uploaded file"
        
        # Count successful classifications
        successful_classifications = sum(1 for r in classification_results if r.get('method') != 'error')
        
        if successful_classifications == len(extracted_figures):
            st.success(f"✅ Successfully extracted and classified {len(extracted_figures)} figures {source_info}!")
        else:
            st.warning(f"⚠️ Extracted {len(extracted_figures)} figures, successfully classified {successful_classifications} {source_info}")
        
        # Show classification method summary
        ai_count = sum(1 for r in classification_results if r.get('method') == 'gemini_api')
        heuristic_count = sum(1 for r in classification_results if r.get('method') == 'heuristic')
        
        if ai_count > 0:
            st.info(f"🤖 AI classifications: {ai_count}, 🔧 Heuristic classifications: {heuristic_count}")
        
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass


def validate_pdf_url(url):
    """Validate a PDF URL without downloading."""
    try:
        with st.spinner("Validating URL..."):
            downloader = PDFDownloader()
            file_info = downloader.get_file_info_from_url(url)

            if file_info is None:
                st.error("❌ Could not access the URL. Please check if it's valid and accessible.")
                return

            if not file_info['is_pdf']:
                st.error("❌ The URL does not point to a PDF file.")
                return

            # Display file information
            st.success("✅ Valid PDF URL!")
            st.info(f"""
            **File Information:**
            - Size: {file_info['file_size_mb']} MB
            - Content Type: {file_info['content_type']}
            - URL: {file_info['url']}
            """)

    except Exception as e:
        st.error(f"❌ Error validating URL: {str(e)}")


def process_pdf_from_url(url):
    """Process a PDF file from a URL."""
    try:
        with st.spinner("Downloading PDF from URL..."):
            # Download PDF from URL
            downloader = PDFDownloader()
            tmp_file_path = downloader.download_pdf_from_url(url)

            # Create a mock uploaded file object for compatibility
            with open(tmp_file_path, 'rb') as f:
                pdf_content = f.read()

            class MockUploadedFile:
                def __init__(self, content, name):
                    self.content = content
                    self.name = name

                def getvalue(self):
                    return self.content

            mock_file = MockUploadedFile(pdf_content, url.split('/')[-1])

            # Process the downloaded PDF
            process_pdf(mock_file, from_url=True, url=url)

            # Clean up
            os.unlink(tmp_file_path)

    except Exception as e:
        st.error(f"❌ Error processing PDF from URL: {str(e)}")


def display_welcome_screen():
    """Display welcome screen with instructions."""
    st.markdown("""
    ## 🚀 Welcome to FigSense
    
    This AI-powered tool helps you:
    - 📄 Upload PDF documents or provide URLs
    - 🖼️ Extract all figures and images automatically
    - 🤖 Classify figure types using advanced AI
    - 📊 View comprehensive analysis and statistics
    - 💾 Download individual figures or complete ZIP archives
    - 📄 Generate detailed PDF analysis reports
    
    ### 🎯 Supported Figure Types (AI-Powered Classification):
    - **📊 Charts**: Bar charts, pie charts, line graphs, scatter plots, histograms, heatmaps
    - **🔄 Diagrams**: Flowcharts, organizational charts, network diagrams, scientific diagrams
    - **🔧 Technical**: Engineering diagrams, medical diagrams, floor plans
    - **📷 Images**: Photographs, screenshots, logos, infographics
    - **📋 Data**: Tables, timelines, and other data visualizations
    - **🗺️ Maps**: Geographic maps, spatial representations
    
    ### 📋 How to Use:
    
    **Step 1: Configure API Key (Recommended)**
    1. 🔑 Expand the "Gemini API Key" section in the sidebar
    2. 🌐 Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
    3. 📝 Paste your API key and click "Test API Key"
    4. 💾 Click "Save Key" when validation succeeds
    
    **Step 2: Upload or Link Your PDF**
    
    **📁 Option A: Upload from Computer**
    1. Click "Choose a PDF file" in the Upload File tab
    2. Select your PDF document
    3. Click "Process Uploaded PDF" to start extraction
    
    **🔗 Option B: Use URL**
    1. Switch to the "From URL" tab
    2. Enter the direct URL to a PDF file
    3. Click "Validate URL" to verify accessibility
    4. Click "Process PDF" to start extraction
    
    **Step 3: View Results**
    - 📊 See AI-powered classifications with confidence scores
    - 📈 View statistics and figure type distribution
    - 💾 Download individual figures or complete archives
    - 📄 Generate comprehensive analysis reports
    
    ---
    
    ### 🔧 Classification Methods:
    
    **🤖 AI Classification (Recommended)**
    - Uses Google's Gemini AI model
    - High accuracy and detailed descriptions
    - Requires valid API key
    - Provides confidence scores and reasoning
    
    **🔧 Heuristic Classification (Fallback)**
    - Uses image analysis algorithms
    -""")
