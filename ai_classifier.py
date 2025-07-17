import os
import json
import logging
import io
import time
import random
import numpy as np
from PIL import Image
import google.generativeai as genai


class AIFigureClassifier:
    """AI-powered figure classifier using Google Gemini."""

    def __init__(self, api_key=None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No Gemini API key provided.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.confidence_score = 0.0

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
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                prompt = self._create_classification_prompt()

                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img = Image.open(img_buffer).convert("RGB")

                response = self.model.generate_content([prompt, img])

                if response.text and response.text.strip().startswith("{"):
                    try:
                        result = json.loads(response.text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing failed: {e}")
                        return self._fallback_classification(image)

                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]

                    self.confidence_score = result.get('confidence', 0.5)

                    return {
                        'classification': result.get('type', 'unknown'),
                        'confidence': self.confidence_score,
                        'description': result.get('description', 'No description available'),
                        'details': result.get('details', {}),
                        'reasoning': result.get('reasoning', '')
                    }

                self.logger.warning("Empty or invalid response from Gemini.")
                return self._fallback_classification(image)

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"AI classification attempt {attempt + 1} failed: {error_msg}")

                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.info(f"Rate limit hit, retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error("Rate limit exceeded. Fallback activated.")
                        return self._fallback_classification(image)
                elif "400" in error_msg and ("API_KEY_INVALID" in error_msg or "expired" in error_msg.lower()):
                    self.logger.error("Invalid/expired API key.")
                    return self._fallback_classification(image)
                else:
                    return self._fallback_classification(image)

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
        try:
            if image is not None:
                img_array = np.array(image)
                height, width = img_array.shape[:2]
                aspect_ratio = width / height

                if len(img_array.shape) == 3:
                    std_color = np.std(img_array)

                    if aspect_ratio > 2:
                        classification = 'timeline'
                        confidence = 0.6
                        description = 'Wide layout suggests timeline or process flow'
                    elif 0.8 <= aspect_ratio <= 1.2 and std_color > 50:
                        classification = 'chart_other'
                        confidence = 0.5
                        description = 'Square, colorful — likely a chart or graph'
                    elif std_color > 80:
                        classification = 'photograph'
                        confidence = 0.4
                        description = 'High color variance — likely real-world image'
                    else:
                        classification = 'diagram_other'
                        confidence = 0.4
                        description = 'Likely a simple diagram or illustration'
                else:
                    classification = 'diagram_other'
                    confidence = 0.3
                    description = 'Grayscale layout — probably a diagram'

                self.confidence_score = confidence
                return {
                    'classification': classification,
                    'confidence': confidence,
                    'description': description,
                    'details': {
                        'visual_elements': ['basic analysis'],
                        'aspect_ratio': f'{aspect_ratio:.2f}',
                        'analysis_method': 'local fallback'
                    },
                    'reasoning': 'Used local heuristic due to API failure.'
                }

            self.confidence_score = 0.3
            return {
                'classification': 'unknown',
                'confidence': 0.3,
                'description': 'Figure present, classification unavailable',
                'details': {},
                'reasoning': 'Fallback: no API or error occurred.'
            }

        except Exception as e:
            self.confidence_score = 0.2
            return {
                'classification': 'unknown',
                'confidence': 0.2,
                'description': 'Figure detected but fallback failed',
                'details': {},
                'reasoning': f'Fallback error: {str(e)}'
            }

    def get_supported_categories(self):
        return self.figure_categories

    def get_confidence(self):
        return self.confidence_score

    def batch_classify(self, images, progress_callback=None):
        results = []
        total = len(images)

        for i, image in enumerate(images):
            result = self.classify_figure(image)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)

        return results
