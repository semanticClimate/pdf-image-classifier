import os
import json
import logging
import base64
import io
import time
import random
import numpy as np
from PIL import Image
from google import generativeai
from google.generativeai import types

class AIFigureClassifier:
    """AI-powered figure classifier using Google Gemini."""
    
    def __init__(self, api_key=None):
        self.logger = logging.getLogger(__name__)
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No Gemini API key provided. Please provide one via parameter or environment variable.")
        self.client = genai.Client(api_key=self.api_key)
        self.confidence_score = 0.0
        
        # Define comprehensive figure categories
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
    
    def classify_figure(self, image):
        """
        Classify a figure using Google Gemini AI with retry logic.
        
        Args:
            image (PIL.Image): The image to classify
            
        Returns:
            dict: Classification results with type, confidence, and description
        """
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                image_bytes = img_buffer.read()
                
                # Create the classification prompt
                prompt = self._create_classification_prompt()
                
                # Call Gemini API
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png",
                        ),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                
                if response.text:
                    result = json.loads(response.text)
                    # Handle both dict and list responses
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]
                    self.confidence_score = result.get('confidence', 0.5) if isinstance(result, dict) else 0.5
                    
                    return {
                        'classification': result.get('type', 'unknown') if isinstance(result, dict) else 'unknown',
                        'confidence': self.confidence_score,
                        'description': result.get('description', 'No description available') if isinstance(result, dict) else 'No description available',
                        'details': result.get('details', {}) if isinstance(result, dict) else {},
                        'reasoning': result.get('reasoning', '') if isinstance(result, dict) else ''
                    }
                else:
                    return self._fallback_classification(image)
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"AI classification attempt {attempt + 1} failed: {error_msg}")
                
                # Check if it's a rate limit error or expired key
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.info(f"Rate limit hit, waiting {delay:.2f} seconds before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error("Rate limit exceeded, using fallback classification")
                        return self._fallback_classification(image)
                elif "400" in error_msg and ("API_KEY_INVALID" in error_msg or "expired" in error_msg.lower()):
                    # API key expired or invalid
                    self.logger.error("API key expired or invalid, using fallback classification")
                    return self._fallback_classification(image)
                else:
                    # For other errors, use fallback immediately
                    return self._fallback_classification(image)
        
        return self._fallback_classification(image)
    
    def get_confidence(self):
        """Get the confidence score of the last classification."""
        return self.confidence_score
    
    def _create_classification_prompt(self):
        """Create a comprehensive classification prompt."""
        categories_text = "\n".join([f"- {key}: {desc}" for key, desc in self.figure_categories.items()])
        
        prompt = f"""
        Analyze this figure/image and classify it into one of the following categories. Be very precise and accurate.

        AVAILABLE CATEGORIES:
        {categories_text}

        CLASSIFICATION REQUIREMENTS:
        1. Look carefully at the visual elements, structure, and content
        2. Consider the purpose and typical use of the figure
        3. For charts/graphs, identify the specific type (bar, pie, line, scatter, etc.)
        4. For diagrams, determine the specific domain (scientific, medical, engineering, etc.)
        5. For images, distinguish between photographs, screenshots, logos, etc.

        SPECIAL CONSIDERATIONS:
        - Tables: Look for structured data in rows and columns
        - Charts: Identify data visualization patterns (bars, lines, circles, points)
        - Diagrams: Look for flowcharts, organizational structures, technical drawings
        - Scientific: Look for formulas, molecular structures, anatomical drawings
        - Maps: Geographic features, roads, boundaries, topographical elements

        OUTPUT FORMAT (JSON):
        {{
            "type": "category_key_from_list_above",
            "confidence": 0.95,
            "description": "Brief description of what you see",
            "details": {{
                "visual_elements": ["list", "of", "key", "elements"],
                "data_type": "type of data shown if applicable",
                "domain": "subject domain if applicable"
            }},
            "reasoning": "Why you chose this classification"
        }}

        Be extremely accurate. If you're not sure between two categories, pick the most specific one that fits best.
        """
        return prompt
    
    def _fallback_classification(self, image=None):
        """Enhanced fallback classification with basic visual analysis."""
        try:
            if image is not None:
                # Basic visual analysis
                img_array = np.array(image)
                height, width = img_array.shape[:2]
                aspect_ratio = width / height
                
                # Simple heuristics based on visual properties
                if len(img_array.shape) == 3:
                    # Color image
                    mean_color = np.mean(img_array)
                    std_color = np.std(img_array)
                    
                    # Simple classification logic
                    if aspect_ratio > 2:
                        classification = 'timeline'
                        confidence = 0.6
                        description = 'Wide horizontal layout suggests timeline or process flow'
                    elif 0.8 <= aspect_ratio <= 1.2 and std_color > 50:
                        classification = 'chart_other'
                        confidence = 0.5
                        description = 'Square format with varied colors suggests chart or graph'
                    elif std_color > 80:
                        classification = 'photograph'
                        confidence = 0.4
                        description = 'High color variation suggests photographic content'
                    else:
                        classification = 'diagram_other'
                        confidence = 0.4
                        description = 'Simple diagram or illustration'
                else:
                    classification = 'diagram_other'
                    confidence = 0.3
                    description = 'Grayscale content, likely diagram or text'
                
                self.confidence_score = confidence
                return {
                    'classification': classification,
                    'confidence': confidence,
                    'description': description,
                    'details': {
                        'visual_elements': ['basic visual analysis'],
                        'analysis_method': 'Local fallback analysis',
                        'aspect_ratio': f'{aspect_ratio:.2f}'
                    },
                    'reasoning': f'AI quota exceeded, used local analysis. Aspect ratio: {aspect_ratio:.2f}'
                }
            else:
                self.confidence_score = 0.3
                return {
                    'classification': 'unknown',
                    'confidence': 0.3,
                    'description': 'Figure detected but classification unavailable',
                    'details': {
                        'visual_elements': ['visual content'],
                        'analysis_method': 'Basic fallback'
                    },
                    'reasoning': 'AI classification unavailable due to quota limits'
                }
        except Exception as e:
            self.confidence_score = 0.2
            return {
                'classification': 'unknown',
                'confidence': 0.2,
                'description': 'Figure extraction successful but classification failed',
                'details': {
                    'visual_elements': ['visual content'],
                    'analysis_method': 'Error fallback'
                },
                'reasoning': f'Fallback analysis failed: {str(e)}'
            }
    
    def get_supported_categories(self):
        """Get all supported figure categories."""
        return self.figure_categories
    
    def batch_classify(self, images, progress_callback=None):
        """
        Classify multiple images in batch.
        
        Args:
            images (list): List of PIL Images
            progress_callback (function): Optional callback for progress updates
            
        Returns:
            list: List of classification results
        """
        results = []
        total = len(images)
        
        for i, image in enumerate(images):
            result = self.classify_figure(image)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
