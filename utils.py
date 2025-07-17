import os
import zipfile
import io
from PIL import Image
import streamlit as st

def get_file_size(uploaded_file):
    """Get human-readable file size."""
    size = len(uploaded_file.getvalue())
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def create_download_link(data, filename, text="Download"):
    """Create a download link for data."""
    return f'<a href="data:application/octet-stream;base64,{data}" download="{filename}">{text}</a>'

def resize_image_for_display(image, max_width=400, max_height=400):
    """Resize image for display while maintaining aspect ratio."""
    # Calculate scaling factor
    width_ratio = max_width / image.width
    height_ratio = max_height / image.height
    scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
    
    if scale_factor < 1.0:
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def validate_image(image):
    """Validate that the image is suitable for processing."""
    if image.width < 10 or image.height < 10:
        return False, "Image too small"
    
    if image.width > 5000 or image.height > 5000:
        return False, "Image too large"
    
    return True, "Valid"

def get_image_info(image):
    """Get detailed information about an image."""
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format,
        'size': f"{image.width}x{image.height}",
        'aspect_ratio': round(image.width / image.height, 2)
    }

def create_thumbnail(image, size=(150, 150)):
    """Create a thumbnail of the image."""
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail

def safe_filename(filename):
    """Create a safe filename by removing/replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def format_confidence(confidence):
    """Format confidence score for display."""
    if confidence >= 0.8:
        return f"🟢 {confidence:.1%}"
    elif confidence >= 0.6:
        return f"🟡 {confidence:.1%}"
    else:
        return f"🔴 {confidence:.1%}"

def get_figure_type_emoji(figure_type):
    """Get emoji for figure type."""
    emoji_map = {
        'bar_chart': '📊',
        'pie_chart': '🥧',
        'line_graph': '📈',
        'scatter_plot': '🔍',
        'histogram': '📊',
        'box_plot': '📈',
        'heatmap': '🔥',
        'table': '📋',
        'flowchart': '🔄',
        'organizational_chart': '🏢',
        'network_diagram': '🕸️',
        'scientific_diagram': '🔬',
        'medical_diagram': '🏥',
        'engineering_diagram': '⚙️',
        'map': '🗺️',
        'floor_plan': '🏠',
        'timeline': '⏰',
        'infographic': '📊',
        'photograph': '📷',
        'screenshot': '💻',
        'logo': '🏷️',
        'chart_other': '📊',
        'diagram_other': '📐',
        'diagram': '📐',
        'unknown': '❓'
    }
    return emoji_map.get(figure_type, '📊')

def format_figure_type(figure_type):
    """Format figure type for display."""
    type_map = {
        'bar_chart': 'Bar Chart',
        'pie_chart': 'Pie Chart',
        'line_graph': 'Line Graph',
        'scatter_plot': 'Scatter Plot',
        'histogram': 'Histogram',
        'box_plot': 'Box Plot',
        'heatmap': 'Heatmap',
        'table': 'Table',
        'flowchart': 'Flowchart',
        'organizational_chart': 'Organizational Chart',
        'network_diagram': 'Network Diagram',
        'scientific_diagram': 'Scientific Diagram',
        'medical_diagram': 'Medical Diagram',
        'engineering_diagram': 'Engineering Diagram',
        'map': 'Map',
        'floor_plan': 'Floor Plan',
        'timeline': 'Timeline',
        'infographic': 'Infographic',
        'photograph': 'Photograph',
        'screenshot': 'Screenshot',
        'logo': 'Logo',
        'chart_other': 'Other Chart',
        'diagram_other': 'Other Diagram',
        'diagram': 'Diagram',
        'unknown': 'Unknown'
    }
    return type_map.get(figure_type, figure_type.replace('_', ' ').title())

def calculate_processing_stats(figures, classifications):
    """Calculate processing statistics."""
    if not figures or not classifications:
        return {}
    
    stats = {
        'total_figures': len(figures),
        'unique_types': len(set(c['classification'] for c in classifications)),
        'avg_confidence': sum(c['confidence'] for c in classifications) / len(classifications),
        'total_pages': len(set(c['page'] for c in classifications)),
        'figures_per_page': len(figures) / len(set(c['page'] for c in classifications)),
        'type_distribution': {}
    }
    
    # Calculate type distribution
    for classification in classifications:
        fig_type = classification['classification']
        stats['type_distribution'][fig_type] = stats['type_distribution'].get(fig_type, 0) + 1
    
    return stats

def log_processing_info(pdf_filename, stats):
    """Log processing information."""
    st.sidebar.success(f"""
    **Processing Complete!**
    
    📄 **File:** {pdf_filename}
    📊 **Figures Found:** {stats.get('total_figures', 0)}
    🎯 **Types Identified:** {stats.get('unique_types', 0)}
    📈 **Avg Confidence:** {stats.get('avg_confidence', 0):.1%}
    📑 **Pages Processed:** {stats.get('total_pages', 0)}
    """)
