import numpy as np
from PIL import Image, ImageStat
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

class FigureClassifier:
    """Classify extracted figures into different categories."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = None
        self.scaler = None
        self.confidence_score = 0.0
        self.feature_names = [
            'aspect_ratio', 'brightness', 'contrast', 'edge_density',
            'color_diversity', 'text_ratio', 'line_density', 'circle_ratio',
            'rectangle_ratio', 'symmetry_horizontal', 'symmetry_vertical',
            'dominant_color_count', 'saturation_mean', 'hue_variance'
        ]
        
        # Initialize classifier with basic rules
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the classifier with basic feature-based rules."""
        # For this implementation, we'll use a rule-based approach
        # combined with feature extraction for classification
        pass
    
    def classify_figure(self, image):
        """
        Classify a figure into one of the predefined categories.
        
        Args:
            image (PIL.Image): The image to classify
            
        Returns:
            str: The classification category
        """
        try:
            # Extract features from the image
            features = self._extract_features(image)
            
            # Rule-based classification
            classification = self._rule_based_classification(features, image)
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Error classifying figure: {str(e)}")
            return "unknown"
    
    def get_confidence(self):
        """Get the confidence score of the last classification."""
        return self.confidence_score
    
    def _extract_features(self, image):
        """
        Extract features from an image for classification.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale for some calculations
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Basic image properties
        height, width = gray.shape
        aspect_ratio = width / height
        
        # Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Color analysis (if color image)
        if len(img_array.shape) == 3:
            # Color diversity
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            color_diversity = unique_colors / (height * width)
            
            # Saturation and hue analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation_mean = np.mean(hsv[:, :, 1])
            hue_variance = np.var(hsv[:, :, 0])
            
            # Dominant colors
            dominant_color_count = self._count_dominant_colors(img_array)
        else:
            color_diversity = 0
            saturation_mean = 0
            hue_variance = 0
            dominant_color_count = 1
        
        # Text detection (simple approach)
        text_ratio = self._estimate_text_ratio(gray)
        
        # Shape detection
        line_density = self._detect_lines(gray)
        circle_ratio = self._detect_circles(gray)
        rectangle_ratio = self._detect_rectangles(gray)
        
        # Symmetry analysis
        symmetry_horizontal = self._calculate_symmetry(gray, axis=0)
        symmetry_vertical = self._calculate_symmetry(gray, axis=1)
        
        features = {
            'aspect_ratio': aspect_ratio,
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'color_diversity': color_diversity,
            'text_ratio': text_ratio,
            'line_density': line_density,
            'circle_ratio': circle_ratio,
            'rectangle_ratio': rectangle_ratio,
            'symmetry_horizontal': symmetry_horizontal,
            'symmetry_vertical': symmetry_vertical,
            'dominant_color_count': dominant_color_count,
            'saturation_mean': saturation_mean,
            'hue_variance': hue_variance
        }
        
        return features
    
    def _rule_based_classification(self, features, image):
        """
        Classify figure using rule-based approach.
        
        Args:
            features (dict): Extracted features
            image (PIL.Image): Original image
            
        Returns:
            str: Classification category
        """
        # Initialize confidence
        self.confidence_score = 0.5
        
        # Rule 1: High circle ratio suggests pie chart
        if features['circle_ratio'] > 0.3:
            self.confidence_score = 0.8
            return "pie_chart"
        
        # Rule 2: High rectangle ratio with low text suggests bar chart
        if features['rectangle_ratio'] > 0.4 and features['text_ratio'] < 0.3:
            self.confidence_score = 0.7
            return "bar_chart"
        
        # Rule 3: High line density with low rectangle ratio suggests line graph
        if features['line_density'] > 0.3 and features['rectangle_ratio'] < 0.2:
            self.confidence_score = 0.7
            return "line_graph"
        
        # Rule 4: High text ratio suggests table or text diagram
        if features['text_ratio'] > 0.4:
            self.confidence_score = 0.6
            return "table"
        
        # Rule 5: High edge density with geometric shapes suggests flowchart
        if features['edge_density'] > 0.2 and features['rectangle_ratio'] > 0.2:
            self.confidence_score = 0.6
            return "flowchart"
        
        # Rule 6: Low edge density and high color diversity suggests photograph
        if features['edge_density'] < 0.1 and features['color_diversity'] > 0.1:
            self.confidence_score = 0.6
            return "photograph"
        
        # Rule 7: High symmetry suggests scientific diagram
        if features['symmetry_horizontal'] > 0.7 or features['symmetry_vertical'] > 0.7:
            self.confidence_score = 0.6
            return "scientific_diagram"
        
        # Rule 8: Check for scatter plot patterns
        if self._is_scatter_plot(features, image):
            self.confidence_score = 0.7
            return "scatter_plot"
        
        # Rule 9: Check for map-like characteristics
        if self._is_map_like(features, image):
            self.confidence_score = 0.6
            return "map"
        
        # Default classification
        self.confidence_score = 0.4
        return "diagram"
    
    def _estimate_text_ratio(self, gray_image):
        """Estimate the ratio of text in the image."""
        # Simple text detection based on connected components
        # This is a basic approximation
        
        # Apply morphological operations to find text-like regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_area = 0
        total_area = gray_image.shape[0] * gray_image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text-like characteristics: small height, reasonable width
            if h < 50 and w > 20 and area > 100:
                text_area += area
        
        return text_area / total_area
    
    def _detect_lines(self, gray_image):
        """Detect line density in the image."""
        # Use Hough line detection
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            return len(lines) / (gray_image.shape[0] * gray_image.shape[1] / 10000)
        return 0
    
    def _detect_circles(self, gray_image):
        """Detect circle ratio in the image."""
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            total_circle_area = sum(np.pi * r * r for (x, y, r) in circles)
            total_area = gray_image.shape[0] * gray_image.shape[1]
            return total_circle_area / total_area
        return 0
    
    def _detect_rectangles(self, gray_image):
        """Detect rectangle ratio in the image."""
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangle_area = 0
        total_area = gray_image.shape[0] * gray_image.shape[1]
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If polygon has 4 vertices, it might be a rectangle
            if len(approx) == 4:
                rectangle_area += cv2.contourArea(contour)
        
        return rectangle_area / total_area
    
    def _calculate_symmetry(self, gray_image, axis):
        """Calculate symmetry of the image along specified axis."""
        if axis == 0:  # Horizontal symmetry
            mid = gray_image.shape[0] // 2
            top_half = gray_image[:mid, :]
            bottom_half = np.flip(gray_image[mid:, :], axis=0)
            
            # Make sure both halves have same dimensions
            min_rows = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_rows, :]
            bottom_half = bottom_half[:min_rows, :]
            
        else:  # Vertical symmetry
            mid = gray_image.shape[1] // 2
            left_half = gray_image[:, :mid]
            right_half = np.flip(gray_image[:, mid:], axis=1)
            
            # Make sure both halves have same dimensions
            min_cols = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_cols]
            right_half = right_half[:, :min_cols]
        
        # Calculate correlation
        if axis == 0:
            correlation = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
        else:
            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        
        return max(0, correlation) if not np.isnan(correlation) else 0
    
    def _count_dominant_colors(self, img_array):
        """Count dominant colors in the image."""
        # Reshape image to be a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=min(8, len(np.unique(pixels, axis=0))), random_state=42)
            kmeans.fit(pixels)
            return len(kmeans.cluster_centers_)
        except:
            return 1
    
    def _is_scatter_plot(self, features, image):
        """Check if image looks like a scatter plot."""
        # Convert to grayscale for analysis
        gray = np.array(image.convert('L'))
        
        # Look for point-like structures
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=15, minRadius=1, maxRadius=10)
        
        if circles is not None and len(circles[0]) > 20:
            return True
        return False
    
    def _is_map_like(self, features, image):
        """Check if image looks like a map."""
        # Maps typically have irregular shapes, varied colors, and specific patterns
        return (features['color_diversity'] > 0.05 and 
                features['edge_density'] > 0.1 and 
                features['edge_density'] < 0.3 and
                features['symmetry_horizontal'] < 0.3 and
                features['symmetry_vertical'] < 0.3)
