import numpy as np
import cv2
from PIL import Image
import logging
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ImageFusionModel:
    """
    Lightweight Vision Transformer-inspired Image Fusion Model
    
    This implementation uses attention mechanisms and multi-scale processing
    to fuse satellite images effectively for college project demonstration.
    """
    
    def __init__(self):
        """Initialize the fusion model"""
        self.logger = logging.getLogger(__name__)
        self.patch_size = 16  # ViT-inspired patch size
        self.attention_heads = 4
        
    def preprocess_image(self, image_path):
        """
        Preprocess image for fusion
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Load image using PIL for better format support
            img_pil = Image.open(image_path)
            
            # Convert to RGB if needed
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            
            # Convert to numpy array
            img = np.array(img_pil)
            
            # Resize to standard size for fusion (optional, for consistency)
            target_size = (512, 512)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            self.logger.debug(f"Preprocessed image shape: {img.shape}")
            return img
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def create_patches(self, image):
        """
        Create patches from image (ViT-inspired)
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Image patches
        """
        h, w, c = image.shape
        
        # Ensure image dimensions are divisible by patch_size
        h_pad = (self.patch_size - h % self.patch_size) % self.patch_size
        w_pad = (self.patch_size - w % self.patch_size) % self.patch_size
        
        if h_pad > 0 or w_pad > 0:
            image = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), mode='reflect')
        
        h, w, c = image.shape
        
        # Create patches
        patches = []
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size, :]
                patches.append(patch.flatten())
        
        return np.array(patches)
    
    def attention_mechanism(self, patches_list):
        """
        Simple attention mechanism for patch fusion
        
        Args:
            patches_list (list): List of patch arrays from different images
            
        Returns:
            numpy.ndarray: Attention-weighted patches
        """
        num_images = len(patches_list)
        num_patches, patch_dim = patches_list[0].shape
        
        # Compute attention weights based on patch variance (texture information)
        attention_weights = np.zeros((num_images, num_patches))
        
        for i, patches in enumerate(patches_list):
            for j in range(num_patches):
                # Use variance as attention score (higher variance = more detail)
                attention_weights[i, j] = np.var(patches[j])
        
        # Normalize attention weights
        attention_weights = attention_weights / (np.sum(attention_weights, axis=0, keepdims=True) + 1e-8)
        
        # Apply attention weights
        fused_patches = np.zeros_like(patches_list[0])
        for i in range(num_images):
            for j in range(num_patches):
                fused_patches[j] += attention_weights[i, j] * patches_list[i][j]
        
        return fused_patches
    
    def multiscale_fusion(self, images):
        """
        Perform multiscale fusion using Laplacian pyramids
        
        Args:
            images (list): List of preprocessed images
            
        Returns:
            numpy.ndarray: Fused image
        """
        # Convert to grayscale for pyramid processing
        gray_images = []
        for img in images:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_images.append(gray.astype(np.float32) / 255.0)
        
        # Create Laplacian pyramids
        pyramids = []
        for gray_img in gray_images:
            pyramid = self.build_laplacian_pyramid(gray_img, levels=4)
            pyramids.append(pyramid)
        
        # Fuse pyramids level by level
        fused_pyramid = []
        for level in range(len(pyramids[0])):
            level_images = [pyramid[level] for pyramid in pyramids]
            fused_level = self.fuse_pyramid_level(level_images)
            fused_pyramid.append(fused_level)
        
        # Reconstruct fused image
        fused_gray = self.collapse_laplacian_pyramid(fused_pyramid)
        
        # Apply color mapping from original images
        fused_color = self.apply_color_mapping(fused_gray, images)
        
        return fused_color
    
    def build_laplacian_pyramid(self, image, levels):
        """Build Laplacian pyramid"""
        gaussian_pyramid = [image]
        
        # Build Gaussian pyramid
        for i in range(levels):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image)
        
        # Build Laplacian pyramid
        laplacian_pyramid = []
        for i in range(levels):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            laplacian = gaussian_pyramid[i] - cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian_pyramid.append(laplacian)
        
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid
    
    def fuse_pyramid_level(self, level_images):
        """Fuse images at a pyramid level"""
        # Simple maximum selection based on local variance
        fused = np.zeros_like(level_images[0])
        
        for i in range(level_images[0].shape[0]):
            for j in range(level_images[0].shape[1]):
                # Find pixel with maximum absolute value (assuming high contrast = important detail)
                pixel_values = [img[i, j] for img in level_images]
                max_idx = np.argmax([abs(val) for val in pixel_values])
                fused[i, j] = pixel_values[max_idx]
        
        return fused
    
    def collapse_laplacian_pyramid(self, pyramid):
        """Collapse Laplacian pyramid to reconstruct image"""
        image = pyramid[-1]
        
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            image = cv2.pyrUp(image, dstsize=size) + pyramid[i]
        
        return image
    
    def apply_color_mapping(self, fused_gray, original_images):
        """Apply color information from original images to fused grayscale"""
        # Use the first color image as base and adjust intensity
        base_color = original_images[0]
        
        # Convert base to HSV
        base_hsv = cv2.cvtColor((base_color * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        base_hsv = base_hsv.astype(np.float32) / 255.0
        
        # Replace value channel with fused intensity
        fused_gray_norm = np.clip(fused_gray, 0, 1)
        base_hsv[:, :, 2] = fused_gray_norm
        
        # Convert back to RGB
        fused_color = cv2.cvtColor((base_hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        fused_color = fused_color.astype(np.float32) / 255.0
        
        return fused_color
    
    def calculate_metrics(self, images, fused_image):
        """
        Calculate fusion quality metrics
        
        Args:
            images (list): Original images
            fused_image (numpy.ndarray): Fused image
            
        Returns:
            dict: Fusion quality metrics
        """
        metrics = {}
        
        try:
            # Convert to grayscale for metrics calculation
            gray_fused = cv2.cvtColor((fused_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Entropy (information content)
            hist = cv2.calcHist([gray_fused], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist[hist > 0]  # Remove zero entries
            entropy = -np.sum((hist / hist.sum()) * np.log2(hist / hist.sum() + 1e-10))
            metrics['entropy'] = round(float(entropy), 3)
            
            # Standard deviation (contrast measure)
            std_dev = np.std(gray_fused)
            metrics['contrast'] = round(float(std_dev), 3)
            
            # Average gradient (edge preservation)
            grad_x = cv2.Sobel(gray_fused, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_fused, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            metrics['edge_strength'] = round(float(avg_gradient), 3)
            
            # Spatial frequency
            rf = np.mean(np.diff(gray_fused, axis=1)**2)
            cf = np.mean(np.diff(gray_fused, axis=0)**2)
            spatial_freq = np.sqrt(rf + cf)
            metrics['spatial_frequency'] = round(float(spatial_freq), 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            metrics = {'entropy': 0, 'contrast': 0, 'edge_strength': 0, 'spatial_frequency': 0}
        
        return metrics
    
    def fuse_images(self, image_paths, output_path):
        """
        Main fusion function
        
        Args:
            image_paths (list): List of input image paths
            output_path (str): Path to save fused image
            
        Returns:
            dict: Fusion metrics
        """
        try:
            self.logger.info(f"Starting fusion of {len(image_paths)} images")
            
            # Preprocess images
            images = []
            for path in image_paths:
                img = self.preprocess_image(path)
                images.append(img)
            
            # Perform patch-based attention fusion
            patches_list = []
            for img in images:
                patches = self.create_patches(img)
                patches_list.append(patches)
            
            # Apply attention mechanism
            fused_patches = self.attention_mechanism(patches_list)
            
            # Reconstruct image from patches
            h, w, c = images[0].shape
            patch_h = h // self.patch_size
            patch_w = w // self.patch_size
            
            fused_attention = np.zeros((h, w, c))
            patch_idx = 0
            
            for i in range(0, h, self.patch_size):
                for j in range(0, w, self.patch_size):
                    patch_data = fused_patches[patch_idx].reshape(self.patch_size, self.patch_size, c)
                    fused_attention[i:i+self.patch_size, j:j+self.patch_size, :] = patch_data
                    patch_idx += 1
            
            # Perform multiscale fusion for enhancement
            fused_multiscale = self.multiscale_fusion(images)
            
            # Combine attention and multiscale results
            alpha = 0.7
            final_fused = alpha * fused_attention + (1 - alpha) * fused_multiscale
            
            # Ensure values are in valid range
            final_fused = np.clip(final_fused, 0, 1)
            
            # Convert to uint8 and save
            result_image = (final_fused * 255).astype(np.uint8)
            result_pil = Image.fromarray(result_image)
            result_pil.save(output_path, 'PNG', quality=95)
            
            # Calculate metrics
            metrics = self.calculate_metrics(images, final_fused)
            
            self.logger.info(f"Fusion completed successfully. Output saved to: {output_path}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in fusion process: {str(e)}")
            raise
