"""
Annotation Module for Archaeological Site Detection
Uses traditional computer vision techniques for feature detection and labeling
"""

import numpy as np
import cv2
from skimage import feature, morphology, filters, measure
from skimage.segmentation import watershed
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArchaeologicalFeatureDetector:
    """Traditional computer vision approach for archaeological feature detection"""
    
    def __init__(self, min_feature_size: int = 50, max_feature_size: int = 1000):
        """
        Initialize feature detector
        
        Args:
            min_feature_size: Minimum feature size in pixels
            max_feature_size: Maximum feature size in pixels
        """
        self.min_feature_size = min_feature_size
        self.max_feature_size = max_feature_size
        
        logger.info(f"Initialized feature detector with size range: {min_feature_size}-{max_feature_size} pixels")
    
    def detect_linear_features(self, elevation: np.ndarray, 
                             slope: np.ndarray, 
                             threshold: float = 0.1) -> List[Dict]:
        """
        Detect linear features (roads, boundaries, etc.)
        
        Args:
            elevation: Elevation data
            slope: Slope data
            threshold: Threshold for line detection
            
        Returns:
            List of detected linear features
        """
        # Use edge detection on elevation and slope
        edges_elevation = feature.canny(elevation, sigma=1.0)
        edges_slope = feature.canny(slope, sigma=1.0)
        
        # Combine edges
        combined_edges = np.logical_or(edges_elevation, edges_slope)
        
        # Apply morphological operations to connect broken lines
        kernel = morphology.disk(2)
        connected_edges = morphology.binary_closing(combined_edges, kernel)
        
        # Find line segments using Hough transform
        lines = cv2.HoughLinesP(
            (connected_edges * 255).astype(np.uint8),
            rho=1, theta=np.pi/180, threshold=50,
            minLineLength=20, maxLineGap=10
        )
        
        linear_features = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if self.min_feature_size <= length <= self.max_feature_size:
                    linear_features.append({
                        'type': 'linear_feature',
                        'coordinates': [(x1, y1), (x2, y2)],
                        'length': length,
                        'confidence': min(length / 100, 1.0)  # Normalize confidence
                    })
        
        return linear_features
    
    def detect_circular_features(self, elevation: np.ndarray, 
                               hillshade: np.ndarray,
                               min_radius: int = 10, 
                               max_radius: int = 50) -> List[Dict]:
        """
        Detect circular features (mounds, pits, etc.)
        
        Args:
            elevation: Elevation data
            hillshade: Hillshade data
            min_radius: Minimum radius in pixels
            max_radius: Maximum radius in pixels
            
        Returns:
            List of detected circular features
        """
        # Use gradient magnitude for edge detection
        grad_x = filters.sobel_h(elevation)
        grad_y = filters.sobel_v(elevation)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply threshold to get binary image
        threshold = filters.threshold_otsu(gradient_magnitude)
        binary = gradient_magnitude > threshold
        
        # Find contours
        contours = measure.find_contours(binary, 0.5)
        
        circular_features = []
        for contour in contours:
            # Fit circle to contour
            if len(contour) >= 5:  # Need at least 5 points for circle fitting
                try:
                    # Calculate centroid and radius
                    centroid = np.mean(contour, axis=0)
                    radius = np.mean(np.sqrt(np.sum((contour - centroid)**2, axis=1)))
                    
                    if min_radius <= radius <= max_radius:
                        # Calculate circularity
                        area = len(contour)
                        perimeter = np.sum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
                        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                        
                        if circularity > 0.3:  # Threshold for circularity
                            circular_features.append({
                                'type': 'circular_feature',
                                'center': centroid.tolist(),
                                'radius': radius,
                                'circularity': circularity,
                                'confidence': min(circularity, 1.0)
                            })
                except:
                    continue
        
        return circular_features
    
    def detect_rectangular_features(self, elevation: np.ndarray, 
                                  slope: np.ndarray,
                                  min_area: int = 100,
                                  max_area: int = 5000) -> List[Dict]:
        """
        Detect rectangular features (platforms, buildings, etc.)
        
        Args:
            elevation: Elevation data
            slope: Slope data
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels
            
        Returns:
            List of detected rectangular features
        """
        # Create binary image based on elevation and slope
        # Look for relatively flat areas (low slope) with elevation changes
        flat_areas = slope < np.percentile(slope, 30)
        elevation_changes = np.abs(filters.gaussian(elevation, sigma=1) - elevation) > np.std(elevation) * 0.5
        
        binary = np.logical_and(flat_areas, elevation_changes)
        
        # Apply morphological operations
        kernel = morphology.rectangle(5, 5)
        binary = morphology.binary_opening(binary, kernel)
        binary = morphology.binary_closing(binary, kernel)
        
        # Find connected components
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        rectangular_features = []
        for region in regions:
            if min_area <= region.area <= max_area:
                # Calculate rectangularity
                bbox = region.bbox
                width = bbox[3] - bbox[1]
                height = bbox[2] - bbox[0]
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                
                # Check if shape is roughly rectangular
                if 1.0 <= aspect_ratio <= 3.0:  # Reasonable aspect ratio for rectangles
                    rectangularity = region.area / (width * height) if width * height > 0 else 0
                    
                    if rectangularity > 0.6:  # Threshold for rectangularity
                        rectangular_features.append({
                            'type': 'rectangular_feature',
                            'bbox': bbox,
                            'area': region.area,
                            'aspect_ratio': aspect_ratio,
                            'rectangularity': rectangularity,
                            'confidence': min(rectangularity, 1.0)
                        })
        
        return rectangular_features
    
    def detect_anomalous_elevation(self, elevation: np.ndarray, 
                                 window_size: int = 15,
                                 threshold: float = 2.0) -> List[Dict]:
        """
        Detect anomalous elevation changes
        
        Args:
            elevation: Elevation data
            window_size: Size of sliding window
            threshold: Threshold for anomaly detection (in standard deviations)
            
        Returns:
            List of detected elevation anomalies
        """
        # Apply sliding window to detect local anomalies
        from scipy.ndimage import uniform_filter, generic_filter
        
        # Calculate local mean and standard deviation
        local_mean = uniform_filter(elevation, size=window_size)
        local_var = uniform_filter(elevation**2, size=window_size) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Calculate z-scores
        z_scores = np.abs(elevation - local_mean) / (local_std + 1e-8)
        
        # Find anomalies
        anomalies = z_scores > threshold
        
        # Find connected components of anomalies
        labeled = measure.label(anomalies)
        regions = measure.regionprops(labeled)
        
        elevation_anomalies = []
        for region in regions:
            if self.min_feature_size <= region.area <= self.max_feature_size:
                # Calculate anomaly strength
                anomaly_strength = np.mean(z_scores[region.coords[:, 0], region.coords[:, 1]])
                
                elevation_anomalies.append({
                    'type': 'elevation_anomaly',
                    'bbox': region.bbox,
                    'area': region.area,
                    'anomaly_strength': anomaly_strength,
                    'confidence': min(anomaly_strength / threshold, 1.0)
                })
        
        return elevation_anomalies
    
    def detect_terrace_systems(self, elevation: np.ndarray, 
                             slope: np.ndarray,
                             min_terrace_width: int = 10) -> List[Dict]:
        """
        Detect terrace systems on slopes
        
        Args:
            elevation: Elevation data
            slope: Slope data
            min_terrace_width: Minimum terrace width in pixels
            
        Returns:
            List of detected terrace systems
        """
        # Look for areas with alternating high and low slopes
        # This indicates terraced areas
        
        # Apply gradient to slope to find slope changes
        grad_slope_x = filters.sobel_h(slope)
        grad_slope_y = filters.sobel_v(slope)
        slope_gradient = np.sqrt(grad_slope_x**2 + grad_slope_y**2)
        
        # Find areas with high slope gradient (terrace edges)
        terrace_edges = slope_gradient > np.percentile(slope_gradient, 80)
        
        # Apply morphological operations to connect terrace edges
        kernel = morphology.disk(3)
        connected_edges = morphology.binary_closing(terrace_edges, kernel)
        
        # Find connected components
        labeled = measure.label(connected_edges)
        regions = measure.regionprops(labeled)
        
        terrace_systems = []
        for region in regions:
            if region.area >= min_terrace_width * 10:  # Minimum area for terrace system
                # Check if region has multiple parallel lines (characteristic of terraces)
                bbox = region.bbox
                height = bbox[2] - bbox[0]
                width = bbox[3] - bbox[1]
                
                # Calculate orientation and check for parallel structure
                orientation = region.orientation
                
                terrace_systems.append({
                    'type': 'terrace_system',
                    'bbox': bbox,
                    'area': region.area,
                    'orientation': orientation,
                    'confidence': min(region.area / (min_terrace_width * 100), 1.0)
                })
        
        return terrace_systems
    
    def detect_features(self, elevation: np.ndarray, slope: np.ndarray, 
                       hillshade: np.ndarray, curvature: np.ndarray) -> Dict[str, List]:
        """
        Detect all types of archaeological features in a tile
        
        Args:
            elevation: Elevation data
            slope: Slope data
            hillshade: Hillshade data
            curvature: Curvature data
            
        Returns:
            Dictionary containing all detected features
        """
        features = {
            "linear_features": self.detect_linear_features(elevation, slope),
            "circular_features": self.detect_circular_features(elevation, hillshade),
            "rectangular_features": self.detect_rectangular_features(elevation, slope),
            "elevation_anomalies": self.detect_anomalous_elevation(elevation),
            "terrace_systems": self.detect_terrace_systems(elevation, slope)
        }
        
        return features
    
    def analyze_tile(self, tile_path: str) -> Dict[str, Any]:
        """
        Analyze a single tile for archaeological features
        
        Args:
            tile_path: Path to processed tile file
            
        Returns:
            Dictionary containing analysis results
        """
        # Load processed tile data
        tile_name = os.path.splitext(os.path.basename(tile_path))[0]
        
        # Load the processed data
        processed_dir = "data/processed"
        tile_data_path = os.path.join(processed_dir, tile_name)
        
        if not os.path.exists(tile_data_path):
            raise FileNotFoundError(f"Processed data not found for {tile_name}")
        
        # Load individual arrays
        elevation = np.load(os.path.join(tile_data_path, "elevation.npy"))
        slope = np.load(os.path.join(tile_data_path, "slope.npy"))
        aspect = np.load(os.path.join(tile_data_path, "aspect.npy"))
        hillshade = np.load(os.path.join(tile_data_path, "hillshade.npy"))
        curvature = np.load(os.path.join(tile_data_path, "curvature.npy"))
        
        # Perform feature detection
        features = self.detect_features(elevation, slope, hillshade, curvature)
        
        # Create analysis result
        analysis = {
            "tile_name": tile_name,
            "tile_path": tile_path,
            "features": features,
            "statistics": {
                "elevation_mean": float(np.mean(elevation)),
                "elevation_std": float(np.std(elevation)),
                "slope_mean": float(np.mean(slope)),
                "slope_std": float(np.std(slope)),
                "curvature_mean": float(np.mean(curvature)),
                "curvature_std": float(np.std(curvature))
            },
            "feature_counts": {
                "linear_features": len(features.get("linear_features", [])),
                "circular_features": len(features.get("circular_features", [])),
                "rectangular_features": len(features.get("rectangular_features", [])),
                "elevation_anomalies": len(features.get("elevation_anomalies", [])),
                "terrace_systems": len(features.get("terrace_systems", []))
            }
        }
        
        return analysis
    
    def create_visualization(self, elevation: np.ndarray, 
                           slope: np.ndarray, 
                           hillshade: np.ndarray,
                           features: List[Dict],
                           save_path: Optional[str] = None) -> str:
        """
        Create visualization of detected features
        
        Args:
            elevation: Elevation data
            slope: Slope data
            hillshade: Hillshade data
            features: Detected features
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original data
        axes[0, 0].imshow(elevation, cmap='terrain')
        axes[0, 0].set_title('Elevation')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(slope, cmap='hot')
        axes[0, 1].set_title('Slope')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(hillshade, cmap='gray')
        axes[0, 2].set_title('Hillshade')
        axes[0, 2].axis('off')
        
        # Feature overlays
        axes[1, 0].imshow(elevation, cmap='terrain')
        self._plot_features(axes[1, 0], features, 'linear_feature', 'red')
        axes[1, 0].set_title('Linear Features')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(elevation, cmap='terrain')
        self._plot_features(axes[1, 1], features, 'circular_feature', 'blue')
        axes[1, 1].set_title('Circular Features')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(elevation, cmap='terrain')
        self._plot_features(axes[1, 2], features, 'rectangular_feature', 'green')
        axes[1, 2].set_title('Rectangular Features')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            # Save to temporary file
            temp_path = "temp_feature_viz.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
    
    def _plot_features(self, ax, features: List[Dict], feature_type: str, color: str):
        """Helper method to plot features on an axis"""
        for feature in features:
            if feature['type'] == feature_type:
                if feature_type == 'linear_feature':
                    coords = feature['coordinates']
                    ax.plot([coords[0][0], coords[1][0]], 
                           [coords[0][1], coords[1][1]], 
                           color=color, linewidth=2)
                elif feature_type == 'circular_feature':
                    center = feature['center']
                    radius = feature['radius']
                    circle = plt.Circle(center, radius, color=color, fill=False, linewidth=2)
                    ax.add_patch(circle)
                elif feature_type == 'rectangular_feature':
                    bbox = feature['bbox']
                    rect = plt.Rectangle((bbox[1], bbox[0]), 
                                       bbox[3] - bbox[1], bbox[2] - bbox[0],
                                       color=color, fill=False, linewidth=2)
                    ax.add_patch(rect)
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def batch_analyze_tiles(self, tile_paths: List[str], output_dir: str = "analysis") -> List[Dict]:
        """
        Analyze multiple tiles for archaeological features
        
        Args:
            tile_paths: List of paths to processed tile files
            output_dir: Output directory for analysis results
            
        Returns:
            List of analysis results for each tile
        """
        os.makedirs(output_dir, exist_ok=True)
        analyses = []
        
        for tile_path in tile_paths:
            try:
                tile_name = os.path.splitext(os.path.basename(tile_path))[0]
                logger.info(f"Analyzing tile: {tile_name}")
                
                # Analyze tile
                analysis = self.analyze_tile(tile_path)
                
                # Convert NumPy types for JSON serialization
                analysis = self._convert_numpy_types(analysis)
                
                # Save analysis to file
                analysis_path = os.path.join(output_dir, f"{tile_name}_analysis.json")
                with open(analysis_path, 'w') as f:
                    json.dump(analysis, f, indent=2)
                
                analyses.append(analysis)
                logger.info(f"Analysis saved: {analysis_path}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {tile_path}: {e}")
                analyses.append({"error": str(e)})
        
        return analyses
    
    def create_analysis_report(self, analyses: List[Dict[str, Any]], 
                             output_path: str = "analysis/report.html") -> str:
        """
        Create an HTML report of all analyses
        
        Args:
            analyses: List of analysis results
            output_path: Path to save HTML report
            
        Returns:
            Path to saved report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create summary statistics
        total_tiles = len(analyses)
        tiles_with_features = sum(1 for a in analyses if a.get('total_features', 0) > 0)
        avg_confidence = np.mean([a.get('average_confidence', 0) for a in analyses])
        
        # Count assessment types
        assessment_counts = {}
        for analysis in analyses:
            assessment = analysis.get('overall_assessment', 'unknown')
            assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Archaeological Site Detection - Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .tile {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .feature {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .confidence-high {{ color: green; }}
                .confidence-medium {{ color: orange; }}
                .confidence-low {{ color: red; }}
                .assessment-high {{ background-color: #d4edda; }}
                .assessment-moderate {{ background-color: #fff3cd; }}
                .assessment-low {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>Archaeological Site Detection - Analysis Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tiles Analyzed:</strong> {total_tiles}</p>
                <p><strong>Tiles with Features:</strong> {tiles_with_features}</p>
                <p><strong>Average Confidence Score:</strong> {avg_confidence:.2f}</p>
                
                <h3>Assessment Distribution:</h3>
                <ul>
        """
        
        for assessment, count in assessment_counts.items():
            html_content += f"<li><strong>{assessment}:</strong> {count} tiles</li>"
        
        html_content += """
                </ul>
            </div>
            
            <h2>Detailed Analyses</h2>
        """
        
        for analysis in analyses:
            tile_id = analysis.get('tile_id', 'unknown')
            total_features = analysis.get('total_features', 0)
            avg_confidence = analysis.get('average_confidence', 0)
            assessment = analysis.get('overall_assessment', 'unknown')
            features = analysis.get('features', {})
            feature_types = analysis.get('feature_types', {})
            
            confidence_class = 'confidence-high' if avg_confidence > 0.6 else 'confidence-medium' if avg_confidence > 0.3 else 'confidence-low'
            assessment_class = f'assessment-{assessment.split("_")[0]}' if '_' in assessment else 'assessment-low'
            
            html_content += f"""
            <div class="tile {assessment_class}">
                <h3>Tile: {tile_id}</h3>
                <p><strong>Assessment:</strong> {assessment}</p>
                <p><strong>Confidence Score:</strong> <span class="{confidence_class}">{avg_confidence:.2f}</span></p>
                <p><strong>Total Features:</strong> {total_features}</p>
                
                <h4>Feature Types:</h4>
                <ul>
            """
            
            for feature_type, count in feature_types.items():
                html_content += f"<li><strong>{feature_type}:</strong> {count}</li>"
            
            html_content += """
                </ul>
                
                <h4>Detected Features:</h4>
            """
            
            for feature in features.values():
                for f in feature:
                    html_content += f"""
                    <div class="feature">
                        <strong>Type:</strong> {f.get('type', 'unknown')}<br>
                        <strong>Confidence:</strong> {f.get('confidence', 0):.2f}<br>
                        <strong>Details:</strong> {str(f)}
                    </div>
                    """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Analysis report saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    detector = ArchaeologicalFeatureDetector()
    
    # Example tile data (you would load this from your processed tiles)
    example_elevation = np.random.rand(100, 100) * 200 + 100  # 100-300m elevation
    example_slope = np.random.rand(100, 100) * 10  # 0-10 degree slope
    example_hillshade = np.random.rand(100, 100) * 255  # 0-255 hillshade
    
    tile_info = {
        'tile_id': 'example_tile',
        'region': 'Amazonian',
        'coordinates': '(-60.0, -3.0)'
    }
    
    # Analyze the tile
    analysis = detector.analyze_tile(
        tile_path=example_elevation
    )
    
    print("Analysis completed:")
    print(json.dumps(analysis, indent=2)) 