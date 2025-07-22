import streamlit as st
import os
import tempfile
import zipfile
import io
from datetime import datetime
from PIL import Image
import altair as alt
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


def main():
    st.set_page_config(
        page_title="PDF Figure Extraction & Classification Tool",
        page_icon="📊",
        layout="wide")

    st.title("📊 FigSense by Avika Joshi")
    st.markdown(
        "Upload a PDF document to automatically extract and classify all figures within it."
    )
    st.markdown("For queries contact me at contactavikajoshi@gmail.com")

    # Sidebar for file upload and API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input section
        with st.expander("🔑 Gemini API Key", expanded=False):
            st.markdown("Provide your own Gemini API key for AI classification:")
            user_api_key = st.text_input(
                "API Key", 
                type="password",
                placeholder="Enter your Gemini API key...",
                help="Get your API key from https://aistudio.google.com/app/apikey"
            )
            if user_api_key:
                st.session_state.user_api_key = user_api_key
                st.success("✅ API key provided!")
            elif 'user_api_key' not in st.session_state:
                st.info("Using default API key (may have rate limits)")
        
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
                if st.button("Validate URL", type="secondary"):
                    validate_pdf_url(pdf_url)

                if st.button("Process PDF from URL", type="primary"):
                    process_pdf_from_url(pdf_url)

    # Main content area
    if st.session_state.processing_complete and st.session_state.extracted_figures:
        display_results()
    else:
        display_welcome_screen()


def process_pdf(uploaded_file, from_url=False, url=None):
    """Process the uploaded PDF file and extract/classify figures."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Initialize components
        extractor = PDFFigureExtractor()
        
        # Initialize AI classifier with user's API key if provided
        user_api_key = st.session_state.get('user_api_key')
        try:
            classifier = AIFigureClassifier(api_key=user_api_key)
            if user_api_key:
                st.info("🔑 Using your provided API key for classification")
        except ValueError as e:
            st.error(f"API key error: {str(e)}")
            return

        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Extract figures
        status_text.text("Extracting figures from PDF...")
        progress_bar.progress(25)

        extracted_figures = extractor.extract_figures(tmp_file_path)

        if not extracted_figures:
            st.warning("No figures found in the PDF document.")
            os.unlink(tmp_file_path)
            return

        progress_bar.progress(50)
        status_text.text(
            f"Found {len(extracted_figures)} figures. Classifying...")

        # Classify figures using AI with rate limiting protection
        classification_results = []
        for i, figure_data in enumerate(extracted_figures):
            status_text.text(
                f"Classifying figure {i + 1}/{len(extracted_figures)} using AI..."
            )

            classification_result = classifier.classify_figure(
                figure_data['image'])
            classification_results.append({
                'figure_id':
                i,
                'classification':
                classification_result['classification'],
                'confidence':
                classification_result['confidence'],
                'description':
                classification_result['description'],
                'details':
                classification_result.get('details', {}),
                'reasoning':
                classification_result.get('reasoning', ''),
                'page':
                figure_data['page'],
                'bbox':
                figure_data['bbox']
            })

            # Update progress
            progress = 50 + (i + 1) * 40 / len(extracted_figures)
            progress_bar.progress(int(progress))

        progress_bar.progress(100)
        status_text.text("Processing complete!")

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

        source_info = f"from URL: {url}" if from_url else "from uploaded file"
        st.success(
            f"Successfully extracted and classified {len(extracted_figures)} figures {source_info}!"
        )
        st.rerun()

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass


def validate_pdf_url(url):
    """Validate a PDF URL without downloading."""
    try:
        downloader = PDFDownloader()
        file_info = downloader.get_file_info_from_url(url)

        if file_info is None:
            st.error(
                "Could not access the URL. Please check if it's valid and accessible."
            )
            return

        if not file_info['is_pdf']:
            st.error("The URL does not point to a PDF file.")
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
        st.error(f"Error validating URL: {str(e)}")


def process_pdf_from_url(url):
    """Process a PDF file from a URL."""
    try:
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
        st.error(f"Error processing PDF from URL: {str(e)}")


def display_welcome_screen():
    """Display welcome screen with instructions."""
    st.markdown("""
    ## Welcome to the FigSense
    
    This tool helps you:
    - 📄 Upload PDF documents
    - 🖼️ Extract all figures and images
    - 🔍 Classify figure types automatically
    - 📊 View comprehensive analysis
    - 💾 Download individual figures or all as ZIP
    - 📄 Generate detailed PDF analysis reports
    
    ### Supported Figure Types (AI-Powered Classification):
    - **Charts**: Bar charts, pie charts, line graphs, scatter plots, histograms, heatmaps
    - **Diagrams**: Flowcharts, organizational charts, network diagrams, scientific diagrams
    - **Technical**: Engineering diagrams, medical diagrams, floor plans
    - **Images**: Photographs, screenshots, logos, infographics
    - **Data**: Tables, timelines, and other data visualizations
    - **Maps**: Geographic maps, spatial representations
    
    ### How to Use:
    **Step 1: API Key (Optional but Recommended)**
    1. Expand the "Gemini API Key" section in the sidebar
    2. Enter your own API key for better rate limits
    3. Get a free key from: https://aistudio.google.com/app/apikey
    
    **Step 2: Upload PDF**
    **Option A: Upload from Computer**
    1. Click "Choose a PDF file" in the Upload File tab
    2. Select your PDF document
    3. Click "Process Uploaded PDF" to start extraction
    
    **Option B: Use URL**
    1. Switch to the "From URL" tab
    2. Enter the direct URL to a PDF file
    3. Click "Validate URL" to check if it's valid
    4. Click "Process PDF from URL" to start extraction
    
    **After Processing:**
    - View results with AI-powered classifications
    - See detailed descriptions and confidence scores
    - Download individual figures or all as ZIP
    - Generate comprehensive PDF analysis report
    
    Get started by uploading a PDF file!
    """)


def display_results():
    """Display the extracted figures and classification results."""
    figures = st.session_state.extracted_figures
    classifications = st.session_state.classification_results

    # Statistics summary
    st.header("📈 Analysis Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Figures", len(figures))
    with col2:
        unique_types = len(set(c['classification'] for c in classifications))
        st.metric("Figure Types", unique_types)
    with col3:
        avg_confidence = sum(c['confidence']
                             for c in classifications) / len(classifications)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    # Figure type distribution
    st.subheader("Figure Type Distribution")
    type_counts = {}
    for classification in classifications:
        fig_type = classification['classification']
        type_counts[fig_type] = type_counts.get(fig_type, 0) + 1

    # Create distribution chart
    # Assuming type_counts is a dictionary like {'bar_chart': 1, 'diagram_other': 1, 'timeline': 4}
    df_types = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])

    # Altair chart with horizontal x-axis labels
    chart = alt.Chart(df_types).mark_bar().encode(
        x=alt.X('Type:N', axis=alt.Axis(labelAngle=0)),  # 👈 This makes labels horizontal
        y='Count:Q'
    ).properties(
        title='')
    st.altair_chart(chart, use_container_width=True)


    # Download options
    st.subheader("Download Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📦 Download All Figures as ZIP"):
            zip_buffer = create_zip_download(figures, classifications)
            st.download_button(label="Download ZIP",
                               data=zip_buffer,
                               file_name="extracted_figures.zip",
                               mime="application/zip")

    with col2:
        if st.button("📄 Generate Analysis Report"):
            generate_pdf_report(figures, classifications)

    # Display individual figures
    st.header("🖼️ Extracted Figures")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox("Filter by type:",
                                   ['All'] + sorted(list(type_counts.keys())))
    with col2:
        sort_by = st.selectbox("Sort by:",
                               ['Page Number', 'Confidence', 'Figure Type'])

    # Filter and sort figures
    filtered_results = classifications
    if filter_type != 'All':
        filtered_results = [
            c for c in classifications if c['classification'] == filter_type
        ]

    if sort_by == 'Page Number':
        filtered_results.sort(key=lambda x: x['page'])
    elif sort_by == 'Confidence':
        filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_by == 'Figure Type':
        filtered_results.sort(key=lambda x: x['classification'])

    # Display figures in grid
    cols_per_row = 2
    for i in range(0, len(filtered_results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(filtered_results):
                result = filtered_results[i + j]
                figure_data = figures[result['figure_id']]

                with cols[j]:
                    display_figure_card(figure_data, result)


def display_figure_card(figure_data, classification_result):
    """Display a single figure card with classification info."""
    with st.container():
        # Get emoji and formatted name for the figure type
        emoji = get_figure_type_emoji(classification_result['classification'])
        formatted_type = format_figure_type(
            classification_result['classification'])

        st.image(
            figure_data['image'],
            caption=
            f"Page {classification_result['page']} - {emoji} {formatted_type}",
            use_container_width=True)

        # Figure details with enhanced information
        confidence_color = "🟢" if classification_result[
            'confidence'] > 0.8 else "🟡" if classification_result[
                'confidence'] > 0.6 else "🔴"

        st.markdown(f"""
        **Type:** {emoji} {formatted_type}  
        **Confidence:** {confidence_color} {classification_result['confidence']:.1%}  
        **Page:** {classification_result['page']}  
        **Description:** {classification_result.get('description', 'No description available')}
        """)

        # Show additional details if available
        if classification_result.get('details'):
            details = classification_result['details']
            if details.get('visual_elements'):
                st.caption(
                    f"Visual Elements: {', '.join(details['visual_elements'])}"
                )

        # Expandable section for AI reasoning
        if classification_result.get('reasoning'):
            with st.expander("AI Classification Reasoning"):
                st.write(classification_result['reasoning'])

        # Download button for individual figure
        img_buffer = io.BytesIO()
        figure_data['image'].save(img_buffer, format='PNG')
        img_buffer.seek(0)

        st.download_button(
            label="Download Figure",
            data=img_buffer,
            file_name=
            f"figure_page_{classification_result['page']}_{classification_result['figure_id']}.png",
            mime="image/png",
            key=f"download_{classification_result['figure_id']}")


def create_zip_download(figures, classifications):
    """Create a ZIP file containing all extracted figures."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create summary CSV
        summary_data = []
        for i, (figure,
                classification) in enumerate(zip(figures, classifications)):
            summary_data.append({
                'Figure ID': i,
                'Filename': f"figure_page_{classification['page']}_{i}.png",
                'Type': classification['classification'],
                'Confidence': f"{classification['confidence']:.1%}",
                'Page': classification['page']
            })

        df_summary = pd.DataFrame(summary_data)
        csv_buffer = io.StringIO()
        df_summary.to_csv(csv_buffer, index=False)
        zip_file.writestr('figure_summary.csv', csv_buffer.getvalue())

        # Add individual figures
        for i, (figure,
                classification) in enumerate(zip(figures, classifications)):
            img_buffer = io.BytesIO()
            figure['image'].save(img_buffer, format='PNG')
            img_buffer.seek(0)

            filename = f"figure_page_{classification['page']}_{i}.png"
            zip_file.writestr(filename, img_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer


def generate_pdf_report(figures, classifications):
    """Generate and provide download for PDF analysis report."""
    progress_bar = None
    status_text = None

    try:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Generating comprehensive PDF report...")
        progress_bar.progress(25)

        # Validate input data
        if not figures or not classifications:
            st.error("No figures or classifications found to generate report.")
            return

        # Initialize report generator
        report_generator = PDFReportGenerator()

        progress_bar.progress(50)
        status_text.text("Creating report content...")

        # Generate report with better error handling
        source_info = st.session_state.get('source_info', 'PDF Document')

        try:
            pdf_buffer = report_generator.create_summary_buffer(
                figures, classifications, source_info)
        except Exception as report_error:
            st.error(f"Error creating report content: {str(report_error)}")
            # Try to create a simplified report
            st.warning("Attempting to create simplified report...")
            try:
                # Create minimal report with text only
                from io import BytesIO
                pdf_buffer = BytesIO()
                # Simple fallback: create a basic text summary
                summary_text = f"""
                PDF Figure Analysis Report
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Source: {source_info}
                
                Summary:
                - Total figures extracted: {len(figures)}
                - Total classifications: {len(classifications)}
                
                Figure Types Found:
                """

                type_counts = {}
                for classification in classifications:
                    fig_type = classification['classification']
                    type_counts[fig_type] = type_counts.get(fig_type, 0) + 1

                for fig_type, count in type_counts.items():
                    summary_text += f"- {fig_type}: {count}\n"

                # Create a simple text file as fallback
                pdf_buffer.write(summary_text.encode('utf-8'))
                pdf_buffer.seek(0)

                st.warning(
                    "Created simplified text report due to PDF generation issues."
                )

            except Exception as fallback_error:
                st.error(f"Could not create report: {str(fallback_error)}")
                return

        progress_bar.progress(75)
        status_text.text("Finalizing report...")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figure_analysis_report_{timestamp}.pdf"

        progress_bar.progress(100)
        status_text.text("Report ready for download!")

        # Provide download button
        st.download_button(
            label="📄 Download Analysis Report (PDF)",
            data=pdf_buffer,
            file_name=filename,
            mime="application/pdf",
            help=
            "Complete analysis report with figure thumbnails, statistics, and detailed descriptions"
        )

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        st.success(
            "PDF report generated successfully! Click the download button above."
        )

    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        st.info("Please try again or contact support if the issue persists.")

    finally:
        # Clear progress indicators on any exit
        try:
            if progress_bar is not None:
                progress_bar.empty()
            if status_text is not None:
                status_text.empty()
        except:
            pass


if __name__ == "__main__":
    main()
