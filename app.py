import os
import json
import logging
import io
import time
import random
import numpy as np
from PIL import Image
import google.generativeai as genai
import streamlit as st


class AIFigureClassifier:
    """AI-powered figure classifier using Google Gemini."""

    def __init__(self, api_key=None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        self.confidence_score = 0.0
        self.api_configured = False

        # Try to configure Gemini API
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Test the API with a simple request
                test_response = self.model.generate_content("Test connection")
                self.api_configured = True
                self.logger.info("Gemini API configured successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to configure Gemini API: {str(e)}")
                self.api_configured = False
                if "API_KEY_INVALID" in str(e) or "401" in str(e):
                    raise ValueError(f"Invalid Gemini API key: {str(e)}")
                else:
                    st.warning(f"Gemini API configuration failed: {str(e)}. Using fallback classification.")
        else:
            self.logger.warning("No Gemini API key provided. Using fallback classification only.")
            st.warning("⚠️ No Gemini API key provided. Using basic heuristic classification. For better results, please provide a valid Gemini API key.")

        self.figure_categories = {
            "bar_chart": "Bar Chart - Shows data using rectangular bars",
            "pie_chart": "Pie Chart - Circular chart showing proportions",
            "line_graph": "Line Graph - Shows trends over time or continuous data",
            "scatter_plot": "Scatter Plot - Shows relationship between two variables",
            "histogram": "Histogram - Shows distribution of data",
            "box_plot": "Box Plot - Shows statistical distribution",
            "heatmap": "Heatmap - Shows data intensity with colors",
            "flowchart": "Flowchart - Shows process or workflow",
            "organizational_chart": "Organizational Chart - Shows hierarchy",
            "network_diagram": "Network Diagram - Shows connections between entities",
            "scientific_diagram": "Scientific Diagram - Technical/scientific illustration",
            "medical_diagram": "Medical Diagram - Anatomical or medical illustration",
            "engineering_diagram": "Engineering Diagram - Technical drawing or schematic",
            "map": "Map - Geographic or spatial representation",
            "floor_plan": "Floor Plan - Architectural layout",
            "timeline": "Timeline - Shows events over time",
            "table": "Table - Structured data in rows and columns",
            "infographic": "Infographic - Visual information presentation",
            "photograph": "Photograph - Real-world image",
            "screenshot": "Screenshot - Computer screen capture",
            "logo": "Logo - Brand or company symbol",
            "chart_other": "Other Chart Type - Specialized chart not in main categories",
            "diagram_other": "Other Diagram - General diagram or illustration",
            "unknown": "Unknown - Cannot determine figure type"
        }

    def is_api_available(self):
        """Check if Gemini API is properly configured and available."""
        return self.api_configured and self.model is not None

    def classify_figure(self, image):
        """Classify a figure using Gemini API or fallback to heuristic method."""
        
        # If API is not configured, use fallback immediately
        if not self.is_api_available():
            st.info("🔄 Using heuristic classification (no API key provided)")
            return self._fallback_classification(image)

        # Try Gemini API classification
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                st.info(f"🤖 Using Gemini AI for classification (attempt {attempt + 1})")
                
                prompt = self._create_classification_prompt()

                # Prepare image
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img = Image.open(img_buffer).convert("RGB")

                # Make API request
                response = self.model.generate_content([prompt, img])

                if response.text and response.text.strip().startswith("{"):
                    try:
                        result = json.loads(response.text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing failed: {e}")
                        st.warning("⚠️ API response parsing failed, using fallback")
                        return self._fallback_classification(image)

                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]

                    self.confidence_score = result.get('confidence', 0.5)

                    st.success("✅ AI classification successful!")
                    return {
                        'classification': result.get('type', 'unknown'),
                        'confidence': self.confidence_score,
                        'description': result.get('description', 'No description available'),
                        'details': result.get('details', {}),
                        'reasoning': result.get('reasoning', ''),
                        'method': 'gemini_api'
                    }

                st.warning("⚠️ Empty response from Gemini API, using fallback")
                return self._fallback_classification(image)

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"AI classification attempt {attempt + 1} failed: {error_msg}")
                
                # Handle specific error types
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        st.warning(f"⏳ Rate limit hit, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        st.error("❌ Rate limit exceeded. Using fallback classification.")
                        return self._fallback_classification(image)
                        
                elif "400" in error_msg or "401" in error_msg or "403" in error_msg:
                    st.error("❌ API key invalid or expired. Using fallback classification.")
                    self.api_configured = False  # Disable API for future calls
                    return self._fallback_classification(image)
                    
                elif "SAFETY" in error_msg:
                    st.warning("⚠️ Content filtered by safety settings. Using fallback classification.")
                    return self._fallback_classification(image)
                    
                else:
                    st.warning(f"⚠️ API error: {error_msg}. Using fallback classification.")
                    return self._fallback_classification(image)

        # If all retries failed, use fallback
        st.warning("⚠️ All API attempts failed. Using fallback classification.")
        return self._fallback_classification(image)

    def _create_classification_prompt(self):
        categories_text = "\n".join([f"- {key}: {desc}" for key, desc in self.figure_categories.items()])

        return f"""
        Analyze this figure/image and classify it into one of the following categories.

        AVAILABLE CATEGORIES:
        {categories_text}

        CLASSIFICATION REQUIREMENTS:
        1. Look carefully at visual structure and content.
        2. For graphs: identify if bar/pie/line/scatter etc.
        3. For diagrams: identify domain (scientific, engineering, etc.).
        4. For photos/screenshots: recognize realism or UI elements.

        OUTPUT FORMAT (JSON):
        {{
            "type": "category_key_from_list_above",
            "confidence": 0.95,
            "description": "Brief description of what you see",
            "details": {{
                "visual_elements": ["key", "elements"],
                "data_type": "type of data if any",
                "domain": "subject domain if known"
            }},
            "reasoning": "Why you chose this classification"
        }}
        """

    def _fallback_classification(self, image=None):
        """Enhanced fallback classification using image analysis heuristics."""
        try:
            if image is not None:
                img_array = np.array(image)
                height, width = img_array.shape[:2]
                aspect_ratio = width / height

                # Convert to grayscale for analysis
                if len(img_array.shape) == 3:
                    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = img_array

                # Calculate various metrics
                std_intensity = np.std(gray)
                mean_intensity = np.mean(gray)
                
                # Color analysis if available
                if len(img_array.shape) == 3:
                    std_color = np.std(img_array)
                    color_variance = np.var(img_array, axis=(0, 1)).mean()
                else:
                    std_color = std_intensity
                    color_variance = 0

                # Enhanced heuristic classification
                if aspect_ratio > 3:
                    classification = 'timeline'
                    confidence = 0.7
                    description = 'Very wide layout suggests timeline or horizontal process'
                    
                elif aspect_ratio > 2:
                    classification = 'timeline'
                    confidence = 0.6
                    description = 'Wide layout suggests timeline or process flow'
                    
                elif 0.8 <= aspect_ratio <= 1.2:  # Square-ish
                    if std_color > 80:
                        classification = 'pie_chart'
                        confidence = 0.6
                        description = 'Square, colorful layout suggests pie chart or similar'
                    elif std_color > 50:
                        classification = 'chart_other'
                        confidence = 0.5
                        description = 'Square layout with moderate color variation suggests chart'
                    else:
                        classification = 'diagram_other'
                        confidence = 0.4
                        description = 'Square, simple layout suggests diagram'
                        
                elif aspect_ratio > 1.5:  # Wide
                    if std_color > 60:
                        classification = 'bar_chart'
                        confidence = 0.6
                        description = 'Wide, colorful layout suggests bar chart'
                    else:
                        classification = 'flowchart'
                        confidence = 0.5
                        description = 'Wide layout suggests flowchart or process diagram'
                        
                elif aspect_ratio < 0.7:  # Tall
                    if std_color > 60:
                        classification = 'bar_chart'
                        confidence = 0.5
                        description = 'Tall, colorful layout suggests vertical bar chart'
                    else:
                        classification = 'organizational_chart'
                        confidence = 0.5
                        description = 'Tall layout suggests organizational chart or hierarchy'
                        
                else:  # Regular proportions
                    if std_color > 100:
                        classification = 'photograph'
                        confidence = 0.6
                        description = 'High color variance suggests photograph'
                    elif std_color > 70:
                        classification = 'chart_other'
                        confidence = 0.5
                        description = 'Moderate color variance suggests chart or graph'
                    elif mean_intensity > 200:
                        classification = 'screenshot'
                        confidence = 0.4
                        description = 'High brightness suggests screenshot'
                    else:
                        classification = 'diagram_other'
                        confidence = 0.4
                        description = 'Basic characteristics suggest diagram'

                self.confidence_score = confidence
                return {
                    'classification': classification,
                    'confidence': confidence,
                    'description': description,
                    'details': {
                        'visual_elements': ['heuristic analysis'],
                        'aspect_ratio': f'{aspect_ratio:.2f}',
                        'color_variance': f'{color_variance:.1f}',
                        'intensity_std': f'{std_intensity:.1f}',
                        'analysis_method': 'heuristic fallback'
                    },
                    'reasoning': f'Used heuristic analysis: aspect ratio {aspect_ratio:.2f}, color variance {color_variance:.1f}',
                    'method': 'heuristic'
                }

            # If no image provided
            self.confidence_score = 0.3
            return {
                'classification': 'unknown',
                'confidence': 0.3,
                'description': 'Figure present, classification unavailable',
                'details': {},
                'reasoning': 'No image data available for analysis',
                'method': 'fallback'
            }

        except Exception as e:
            self.confidence_score = 0.2
            return {
                'classification': 'unknown',
                'confidence': 0.2,
                'description': 'Figure detected but analysis failed',
                'details': {},
                'reasoning': f'Fallback error: {str(e)}',
                'method': 'error_fallback'
            }

    def get_supported_categories(self):
        return self.figure_categories

    def get_confidence(self):
        return self.confidence_score

    def batch_classify(self, images, progress_callback=None):
        """Classify multiple images with progress tracking."""
        results = []
        total = len(images)

        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total)
            
            result = self.classify_figure(image)
            results.append(result)

        return results
