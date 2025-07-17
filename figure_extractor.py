import fitz  # PyMuPDF
from PIL import Image
import io
import logging

class PDFFigureExtractor:
    """Extract figures and images from PDF documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_figures(self, pdf_path):
        """
        Extract all figures from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of dictionaries containing figure data
        """
        figures = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract images from page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip if image is too small (likely decorative)
                        if pix.width < 50 or pix.height < 50:
                            pix = None
                            continue
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                        else:  # CMYK
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix_rgb.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            pix_rgb = None
                        
                        # Get image position and size
                        img_rect = page.get_image_rects(img)[0] if page.get_image_rects(img) else None
                        
                        # Store figure data
                        figure_data = {
                            'image': img_pil,
                            'page': page_num + 1,
                            'index': img_index,
                            'bbox': img_rect,
                            'width': pix.width,
                            'height': pix.height,
                            'size': len(img_data)
                        }
                        
                        figures.append(figure_data)
                        
                        # Clean up
                        pix = None
                        
                    except Exception as e:
                        self.logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {str(e)}")
                        continue
                
                # Also extract vector graphics as images
                vector_figures = self._extract_vector_graphics(page, page_num + 1)
                figures.extend(vector_figures)
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        return figures
    
    def _extract_vector_graphics(self, page, page_num):
        """
        Extract vector graphics from a page by rendering them as images.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            list: List of vector graphics as images
        """
        vector_figures = []
        
        try:
            # Get page drawings (vector graphics)
            drawings = page.get_drawings()
            
            if not drawings:
                return vector_figures
            
            # Group drawings by proximity to identify figures
            figure_groups = self._group_drawings_by_proximity(drawings)
            
            for group_idx, drawing_group in enumerate(figure_groups):
                try:
                    # Calculate bounding box for the group
                    min_x = min(d['rect'][0] for d in drawing_group)
                    min_y = min(d['rect'][1] for d in drawing_group)
                    max_x = max(d['rect'][2] for d in drawing_group)
                    max_y = max(d['rect'][3] for d in drawing_group)
                    
                    # Skip if too small
                    if (max_x - min_x) < 50 or (max_y - min_y) < 50:
                        continue
                    
                    # Create clipping rectangle
                    clip_rect = fitz.Rect(min_x, min_y, max_x, max_y)
                    
                    # Render the clipped area as image
                    mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    img_pil = Image.open(io.BytesIO(img_data))
                    
                    # Store vector figure data
                    figure_data = {
                        'image': img_pil,
                        'page': page_num,
                        'index': f"vector_{group_idx}",
                        'bbox': clip_rect,
                        'width': pix.width,
                        'height': pix.height,
                        'size': len(img_data),
                        'type': 'vector'
                    }
                    
                    vector_figures.append(figure_data)
                    
                    # Clean up
                    pix = None
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting vector graphic {group_idx} from page {page_num}: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"Error extracting vector graphics from page {page_num}: {str(e)}")
        
        return vector_figures
    
    def _group_drawings_by_proximity(self, drawings, threshold=50):
        """
        Group drawings by proximity to identify coherent figures.
        
        Args:
            drawings: List of drawing objects
            threshold: Distance threshold for grouping
            
        Returns:
            list: List of drawing groups
        """
        if not drawings:
            return []
        
        groups = []
        used_indices = set()
        
        for i, drawing in enumerate(drawings):
            if i in used_indices:
                continue
            
            current_group = [drawing]
            used_indices.add(i)
            
            # Find nearby drawings
            for j, other_drawing in enumerate(drawings):
                if j in used_indices:
                    continue
                
                if self._are_drawings_close(drawing, other_drawing, threshold):
                    current_group.append(other_drawing)
                    used_indices.add(j)
            
            # Only keep groups with substantial content
            if len(current_group) >= 2:
                groups.append(current_group)
        
        return groups
    
    def _are_drawings_close(self, drawing1, drawing2, threshold):
        """
        Check if two drawings are close enough to be part of the same figure.
        
        Args:
            drawing1, drawing2: Drawing objects
            threshold: Distance threshold
            
        Returns:
            bool: True if drawings are close
        """
        rect1 = drawing1['rect']
        rect2 = drawing2['rect']
        
        # Calculate centers
        center1 = ((rect1[0] + rect1[2]) / 2, (rect1[1] + rect1[3]) / 2)
        center2 = ((rect2[0] + rect2[2]) / 2, (rect2[1] + rect2[3]) / 2)
        
        # Calculate distance
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        
        return distance < threshold
