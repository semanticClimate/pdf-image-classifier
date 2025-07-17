import os
import json
import logging
import base64
import io
import requests
from PIL import Image
import numpy as np
import cv2

class FreeFigureClassifier:
    """Free figure classifier using Hugging Face models and local processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.confidence_score = 0.0
        
        # Free image analysis endpoints (no API key required)
        self.hf_api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
        
        # Define comprehensive figure categories
        self.figure_categories = {
            "bar_chart": ["bar chart", "bar graph", "column chart", "histogram", "bars"],
            "pie_chart": ["pie chart", "pie graph", "circular chart", "donut chart"],
            "line_graph": ["line chart", "line graph", "trend", "curve", "time series"],
            "scatter_plot": ["scatter plot", "scatter chart", "dots", "points", "correlation"],
            "histogram": ["histogram", "distribution", "frequency", "bins"],
            "box_plot": ["box plot", "boxplot", "whisker", "quartile"],
            "heatmap": ["heatmap", "heat map", "intensity", "color map", "gradient"],
            "flowchart": ["flowchart", "flow chart", "process", "workflow", "diagram"],
            "organizational_chart": ["organizational chart", "org chart", "hierarchy", "structure"],
            "network_diagram": ["network", "graph", "nodes", "connections", "tree"],
            "scientific_diagram": ["molecule", "chemical", "formula", "scientific", "laboratory"],
            "medical_diagram": ["anatomy", "medical", "body", "organ", "health"],
            "engineering_diagram": ["circuit", "schematic", "technical", "blueprint", "engineering"],
            "map": ["map", "geographic", "location", "street", "geography", "satellite"],
            "floor_plan": ["floor plan", "blueprint", "layout", "room", "building"],
            "timeline": ["timeline", "chronology", "sequence", "history", "events"],
            "table": ["table", "grid", "rows", "columns", "data", "spreadsheet"],
            "infographic": ["infographic", "information", "visual", "statistics"],
            "photograph": ["photo", "picture", "image", "real", "camera", "scene"],
            "screenshot": ["screenshot", "screen", "interface", "software", "application"],
            "logo": ["logo", "brand", "symbol", "emblem", "company"],
            "chart_other": ["chart", "graph", "visualization", "data"],
            "diagram_other": ["diagram", "illustration", "drawing", "figure"],
            "unknown": ["unclear", "unknown", "indeterminate"]
        }
    
    def classify_figure(self, image):
        """
        Classify a figure using free Hugging Face models.
        
        Args:
            image (PIL.Image): The image to classify
            
        Returns:
            dict: Classification results with type, confidence, and description
        """
        try:
            # First, get image description using BLIP model
            description = self._get_image_description(image)
            
            # Then classify based on description and visual analysis
            classification_result = self._classify_from_description(description, image)
            
            return classification_result
            
        except Exception as e:
            self.logger.error(f"Error in free classification: {str(e)}")
            return self._fallback_classification()
    
    def _get_image_description(self, image):
        """Get image description using Hugging Face BLIP model."""
        try:
            # Convert PIL image to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Try Hugging Face API first (free but may have rate limits)
            try:
                response = requests.post(
                    self.hf_api_url,
                    files={"inputs": img_buffer.getvalue()},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
            except:
                pass
            
            # Fallback to local analysis
            return self._analyze_image_locally(image)
            
        except Exception as e:
            self.logger.warning(f"Error getting image description: {str(e)}")
            return "image with visual elements"
    
    def _analyze_image_locally(self, image):
        """Analyze image locally to generate description."""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Basic image properties
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Color analysis
            if len(img_array.shape) == 3:
                # Count unique colors
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                color_diversity = unique_colors / (height * width)
                
                # Brightness analysis
                brightness = np.mean(img_array)
                
                # Simple shape detection hints
                gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
                edges = self._simple_edge_detection(gray)
                edge_density = np.sum(edges > 0) / (height * width)
            else:
                color_diversity = 0
                brightness = np.mean(img_array)
                edge_density = 0
            
            # Generate description based on analysis
            description_parts = []
            
            if aspect_ratio > 1.5:
                description_parts.append("wide")
            elif aspect_ratio < 0.7:
                description_parts.append("tall")
            
            if color_diversity > 0.1:
                description_parts.append("colorful")
            elif color_diversity < 0.01:
                description_parts.append("simple")
            
            if edge_density > 0.2:
                description_parts.append("detailed diagram")
            elif edge_density > 0.1:
                description_parts.append("chart")
            else:
                description_parts.append("image")
            
            if brightness > 200:
                description_parts.append("bright")
            elif brightness < 100:
                description_parts.append("dark")
            
            return " ".join(description_parts) if description_parts else "visual content"
            
        except Exception as e:
            return "visual content"
    
    def _simple_edge_detection(self, gray_image):
        """Simple edge detection using OpenCV."""
        try:
            # Use OpenCV for edge detection
            gray_uint8 = gray_image.astype(np.uint8)
            edges = cv2.Canny(gray_uint8, 50, 150)
            return edges
        except Exception as e:
            # Fallback to simple gradient-based detection
            try:
                h, w = gray_image.shape
                edges = np.zeros_like(gray_image)
                
                # Simple Sobel-like operator
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        gx = gray_image[i-1, j-1] - gray_image[i-1, j+1] + \
                             2*gray_image[i, j-1] - 2*gray_image[i, j+1] + \
                             gray_image[i+1, j-1] - gray_image[i+1, j+1]
                        
                        gy = gray_image[i-1, j-1] - gray_image[i+1, j-1] + \
                             2*gray_image[i-1, j] - 2*gray_image[i+1, j] + \
                             gray_image[i-1, j+1] - gray_image[i+1, j+1]
                        
                        edges[i, j] = min(255, (gx*gx + gy*gy)**0.5)
                
                return edges
            except:
                return np.zeros_like(gray_image)
    
    def _classify_from_description(self, description, image):
        """Classify figure based on description and image analysis."""
        try:
            description_lower = description.lower()
            
            # Score each category based on keyword matches
            category_scores = {}
            
            for category, keywords in self.figure_categories.items():
                score = 0
                for keyword in keywords:
                    if keyword in description_lower:
                        score += len(keyword)  # Longer matches get higher scores
                
                if score > 0:
                    category_scores[category] = score
            
            # Enhanced analysis based on image properties
            enhanced_scores = self._enhance_classification_with_analysis(image, category_scores)
            
            # Determine best classification
            if enhanced_scores:
                best_category = max(enhanced_scores.items(), key=lambda x: x[1])
                classification = best_category[0]
                confidence = min(0.95, 0.5 + (best_category[1] / 20))  # Scale confidence
                
                # Generate reasoning
                reasoning = f"Classified as {classification} based on visual analysis"
                if description and description != "visual content":
                    reasoning += f" and description: '{description}'"
                
                self.confidence_score = confidence
                
                return {
                    'classification': classification,
                    'confidence': confidence,
                    'description': self._generate_description(classification, description),
                    'details': {
                        'visual_elements': self._extract_visual_elements(description, classification),
                        'analysis_method': 'Free local + HuggingFace analysis'
                    },
                    'reasoning': reasoning
                }
            else:
                return self._classify_by_visual_analysis(image, description)
                
        except Exception as e:
            self.logger.error(f"Error in classification: {str(e)}")
            return self._fallback_classification()
    
    def _enhance_classification_with_analysis(self, image, base_scores):
        """Enhance classification using image analysis."""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Aspect ratio analysis
            aspect_ratio = width / height
            
            # Boost scores based on visual characteristics
            enhanced_scores = base_scores.copy()
            
            # Charts typically have certain aspect ratios
            if 0.8 <= aspect_ratio <= 1.5:
                for chart_type in ['bar_chart', 'pie_chart', 'line_graph', 'scatter_plot']:
                    if chart_type in enhanced_scores:
                        enhanced_scores[chart_type] += 5
            
            # Wide images might be timelines or flowcharts
            if aspect_ratio > 2:
                for wide_type in ['timeline', 'flowchart']:
                    if wide_type in enhanced_scores:
                        enhanced_scores[wide_type] += 8
            
            # Color diversity analysis
            if len(img_array.shape) == 3:
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                color_diversity = unique_colors / (height * width)
                
                # High color diversity suggests photos or complex diagrams
                if color_diversity > 0.1:
                    for colorful_type in ['photograph', 'map', 'infographic']:
                        if colorful_type in enhanced_scores:
                            enhanced_scores[colorful_type] += 6
                
                # Low color diversity suggests simple charts
                if color_diversity < 0.01:
                    for simple_type in ['bar_chart', 'line_graph', 'flowchart']:
                        if simple_type in enhanced_scores:
                            enhanced_scores[simple_type] += 4
            
            return enhanced_scores
            
        except Exception as e:
            return base_scores
    
    def _classify_by_visual_analysis(self, image, description):
        """Classify by visual analysis when no keywords match."""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Default classification based on visual properties
            if len(img_array.shape) == 3:
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                color_diversity = unique_colors / (height * width)
                
                if color_diversity > 0.1:
                    classification = "photograph"
                    confidence = 0.6
                elif aspect_ratio > 2:
                    classification = "timeline"
                    confidence = 0.5
                elif 0.8 <= aspect_ratio <= 1.2:
                    classification = "chart_other"
                    confidence = 0.5
                else:
                    classification = "diagram_other"
                    confidence = 0.4
            else:
                classification = "diagram_other"
                confidence = 0.4
            
            self.confidence_score = confidence
            
            return {
                'classification': classification,
                'confidence': confidence,
                'description': f"Visual analysis suggests this is a {classification.replace('_', ' ')}",
                'details': {
                    'visual_elements': ['visual content'],
                    'analysis_method': 'Visual characteristics analysis'
                },
                'reasoning': f"Classified based on visual properties: aspect ratio {aspect_ratio:.2f}"
            }
            
        except Exception as e:
            return self._fallback_classification()
    
    def _generate_description(self, classification, original_description):
        """Generate a proper description based on classification."""
        type_descriptions = {
            'bar_chart': 'A bar chart showing data with rectangular bars',
            'pie_chart': 'A pie chart displaying data as sectors of a circle',
            'line_graph': 'A line graph showing trends or changes over time',
            'scatter_plot': 'A scatter plot showing the relationship between variables',
            'histogram': 'A histogram showing the distribution of data',
            'box_plot': 'A box plot showing statistical data distribution',
            'heatmap': 'A heatmap showing data intensity with colors',
            'flowchart': 'A flowchart showing a process or workflow',
            'table': 'A table organizing data in rows and columns',
            'photograph': 'A photographic image of real-world content',
            'map': 'A map showing geographical or spatial information',
            'scientific_diagram': 'A scientific diagram or illustration',
            'timeline': 'A timeline showing events in chronological order',
            'infographic': 'An infographic presenting information visually'
        }
        
        base_desc = type_descriptions.get(classification, f"A {classification.replace('_', ' ')}")
        
        if original_description and original_description not in ["visual content", "image with visual elements"]:
            return f"{base_desc}. {original_description}"
        else:
            return base_desc
    
    def _extract_visual_elements(self, description, classification):
        """Extract visual elements from description."""
        elements = []
        
        # Add elements based on classification
        if 'chart' in classification or 'graph' in classification:
            elements.extend(['data visualization', 'axes', 'labels'])
        elif 'diagram' in classification:
            elements.extend(['shapes', 'connections', 'text'])
        elif classification == 'photograph':
            elements.extend(['real objects', 'natural lighting'])
        elif classification == 'table':
            elements.extend(['rows', 'columns', 'grid'])
        
        # Extract from description
        if description:
            if any(word in description.lower() for word in ['color', 'bright', 'dark']):
                elements.append('varied colors')
            if any(word in description.lower() for word in ['text', 'label', 'title']):
                elements.append('text content')
            if any(word in description.lower() for word in ['line', 'curve', 'edge']):
                elements.append('linear elements')
        
        return elements[:5] if elements else ['visual content']
    
    def get_confidence(self):
        """Get the confidence score of the last classification."""
        return self.confidence_score
    
    def _fallback_classification(self):
        """Fallback classification when everything fails."""
        self.confidence_score = 0.3
        return {
            'classification': 'unknown',
            'confidence': 0.3,
            'description': 'Could not classify figure reliably',
            'details': {
                'visual_elements': ['visual content'],
                'analysis_method': 'Fallback classification'
            },
            'reasoning': 'Classification failed, using fallback method'
        }