"""
Preprocessing Module for Archaeological Site Detection
Handles terrain derivative computations, normalization, and feature engineering
"""

import numpy as np
import rasterio
from skimage.transform import resize
from matplotlib.colors import LightSource
from typing import Tuple, Dict, List, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerrainProcessor:
    """Handles terrain derivative computations and preprocessing"""
    
    def __init__(self, target_size: Tuple[int, int] = (100, 100)):
        """
        Initialize terrain processor
        
        Args:
            target_size: Target size for resized tiles (height, width)
        """
        self.target_size = target_size
    
    def compute_slope(self, elevation: np.ndarray, pixel_size: float = 90.0) -> np.ndarray:
        """
        Compute slope from elevation data
        
        Args:
            elevation: Elevation array
            pixel_size: Pixel size in meters
            
        Returns:
            Slope array in degrees
        """
        # Compute gradients
        dy, dx = np.gradient(elevation, pixel_size, pixel_size)
        
        # Compute slope magnitude
        slope_rad = np.sqrt(dx**2 + dy**2)
        
        # Convert to degrees
        slope_deg = np.degrees(np.arctan(slope_rad))
        
        return slope_deg
    
    def compute_aspect(self, elevation: np.ndarray, pixel_size: float = 90.0) -> np.ndarray:
        """
        Compute aspect from elevation data
        
        Args:
            elevation: Elevation array
            pixel_size: Pixel size in meters
            
        Returns:
            Aspect array in degrees (0-360)
        """
        # Compute gradients
        dy, dx = np.gradient(elevation, pixel_size, pixel_size)
        
        # Compute aspect
        aspect_rad = np.arctan2(dy, dx)
        
        # Convert to degrees and normalize to 0-360
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (aspect_deg + 360) % 360
        
        return aspect_deg
    
    def compute_hillshade(self, elevation: np.ndarray, 
                         azimuth: float = 315.0, 
                         altitude: float = 45.0, 
                         pixel_size: float = 90.0) -> np.ndarray:
        """
        Compute hillshade from elevation data
        
        Args:
            elevation: Elevation array
            azimuth: Solar azimuth angle in degrees
            altitude: Solar altitude angle in degrees
            pixel_size: Pixel size in meters
            
        Returns:
            Hillshade array (0-255)
        """
        # Create light source
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        
        # Compute hillshade
        hillshade = ls.hillshade(elevation, vert_exag=1, dx=pixel_size, dy=pixel_size)
        
        return hillshade
    
    def compute_curvature(self, elevation: np.ndarray, pixel_size: float = 90.0) -> np.ndarray:
        """
        Compute curvature from elevation data
        
        Args:
            elevation: Elevation array
            pixel_size: Pixel size in meters
            
        Returns:
            Curvature array
        """
        # Compute second derivatives
        dy, dx = np.gradient(elevation, pixel_size, pixel_size)
        dyy, dxy = np.gradient(dy, pixel_size, pixel_size)
        dxy, dxx = np.gradient(dx, pixel_size, pixel_size)
        
        # Compute curvature
        curvature = (dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / (dx**2 + dy**2 + 1e-8)**1.5
        
        return curvature
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize data using specified method
        
        Args:
            data: Input data array
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized data array
        """
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            normalized = (data - data_min) / (data_max - data_min + 1e-8)
            
        elif method == 'zscore':
            # Z-score normalization
            mean = np.nanmean(data)
            std = np.nanstd(data)
            normalized = (data - mean) / (std + 1e-8)
            
        elif method == 'robust':
            # Robust normalization using percentiles
            p2 = np.nanpercentile(data, 2)
            p98 = np.nanpercentile(data, 98)
            normalized = (data - p2) / (p98 - p2 + 1e-8)
            normalized = np.clip(normalized, 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def resize_tile(self, data: np.ndarray, 
                   method: str = 'edge', 
                   preserve_range: bool = True) -> np.ndarray:
        """
        Resize tile to target size
        
        Args:
            data: Input data array
            method: Resize method ('edge', 'constant', 'reflect', 'symmetric', 'wrap')
            preserve_range: Whether to preserve the data range
            
        Returns:
            Resized data array
        """
        resized = resize(data, self.target_size, mode=method, preserve_range=preserve_range)
        return resized
    
    def create_feature_stack(self, elevation: np.ndarray, 
                           pixel_size: float = 90.0,
                           normalize: bool = True) -> np.ndarray:
        """
        Create a multi-channel feature stack from elevation data
        
        Args:
            elevation: Elevation array
            pixel_size: Pixel size in meters
            normalize: Whether to normalize features
            
        Returns:
            Feature stack array with shape (channels, height, width)
        """
        # Compute terrain derivatives
        slope = self.compute_slope(elevation, pixel_size)
        aspect = self.compute_aspect(elevation, pixel_size)
        hillshade = self.compute_hillshade(elevation, pixel_size=pixel_size)
        curvature = self.compute_curvature(elevation, pixel_size)
        
        # Stack features
        features = [elevation, slope, aspect, hillshade, curvature]
        
        if normalize:
            # Normalize each feature
            features = [self.normalize_data(f) for f in features]
        
        # Stack into multi-channel array
        feature_stack = np.stack(features, axis=0)
        
        return feature_stack
    
    def process_tile(self, tile_path: str, 
                    output_path: Optional[str] = None,
                    save_intermediate: bool = False) -> Dict:
        """
        Process a single tile through the complete pipeline
        
        Args:
            tile_path: Path to input tile
            output_path: Path to save processed tile
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Load tile data
        with rasterio.open(tile_path) as src:
            elevation = src.read(1)
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'nodata': src.nodata
            }
        
        # Handle nodata values
        if src.nodata is not None:
            elevation = np.ma.masked_equal(elevation, src.nodata)
            elevation = elevation.filled(np.nan)
        
        # Resize to target size
        elevation_resized = self.resize_tile(elevation)
        
        # Create feature stack
        feature_stack = self.create_feature_stack(elevation_resized)
        
        # Prepare output
        result = {
            'elevation': elevation_resized,
            'feature_stack': feature_stack,
            'metadata': metadata,
            'tile_path': tile_path
        }
        
        # Save processed tile if output path provided
        if output_path:
            self.save_processed_tile(feature_stack, output_path, metadata)
        
        # Save intermediate results if requested
        if save_intermediate:
            self.save_intermediate_results(tile_path, result)
        
        return result
    
    def save_processed_tile(self, feature_stack: np.ndarray, 
                          output_path: str, metadata: Dict) -> None:
        """
        Save processed tile to disk
        
        Args:
            feature_stack: Feature stack array
            output_path: Output file path
            metadata: Geospatial metadata
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=feature_stack.shape[1],
            width=feature_stack.shape[2],
            count=feature_stack.shape[0],
            dtype=feature_stack.dtype,
            crs=metadata['crs'],
            transform=metadata['transform']
        ) as dst:
            dst.write(feature_stack)
        
        logger.info(f"Saved processed tile: {output_path}")
    
    def save_intermediate_results(self, tile_path: str, result: Dict) -> None:
        """
        Save intermediate processing results
        
        Args:
            tile_path: Original tile path
            result: Processing result dictionary
        """
        base_name = os.path.splitext(os.path.basename(tile_path))[0]
        output_dir = f"data/processed/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual features
        feature_names = ['elevation', 'slope', 'aspect', 'hillshade', 'curvature']
        for i, name in enumerate(feature_names):
            feature_path = os.path.join(output_dir, f"{name}.npy")
            np.save(feature_path, result['feature_stack'][i])
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.npy")
        np.save(metadata_path, result['metadata'])
        
        logger.info(f"Saved intermediate results: {output_dir}")
    
    def batch_process_tiles(self, tile_dir: str, 
                          output_dir: str = "data/processed") -> List[Dict]:
        """
        Process multiple tiles in batch
        
        Args:
            tile_dir: Directory containing input tiles
            output_dir: Output directory for processed tiles
            
        Returns:
            List of processing results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all tile files
        tile_files = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
        tile_files.sort()
        
        results = []
        for tile_file in tile_files:
            tile_path = os.path.join(tile_dir, tile_file)
            base_name = os.path.splitext(tile_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}_processed.tif")
            
            logger.info(f"Processing tile: {tile_file}")
            result = self.process_tile(tile_path, output_path)
            results.append(result)
        
        logger.info(f"Processed {len(results)} tiles")
        return results


def create_archaeological_features(elevation: np.ndarray, 
                                 pixel_size: float = 90.0) -> Dict[str, np.ndarray]:
    """
    Create archaeological-specific features
    
    Args:
        elevation: Elevation array
        pixel_size: Pixel size in meters
        
    Returns:
        Dictionary of archaeological features
    """
    processor = TerrainProcessor()
    
    # Basic terrain features
    slope = processor.compute_slope(elevation, pixel_size)
    aspect = processor.compute_aspect(elevation, pixel_size)
    hillshade = processor.compute_hillshade(elevation, pixel_size=pixel_size)
    curvature = processor.compute_curvature(elevation, pixel_size)
    
    # Archaeological-specific features
    # 1. Elevation zones (common for archaeological sites)
    elevation_zones = np.zeros_like(elevation)
    elevation_zones[(elevation >= 50) & (elevation < 150)] = 1  # Low
    elevation_zones[(elevation >= 150) & (elevation < 300)] = 2  # Mid
    elevation_zones[(elevation >= 300)] = 3  # High
    
    # 2. Slope stability (archaeological sites often on stable slopes)
    slope_stability = np.where(slope < 5, 1, 0)  # Very stable
    slope_stability += np.where((slope >= 5) & (slope < 15), 0.5, 0)  # Moderately stable
    
    # 3. Aspect preference (sites often face certain directions)
    aspect_preference = np.where((aspect >= 45) & (aspect <= 135), 1, 0)  # East-facing
    aspect_preference += np.where((aspect >= 225) & (aspect <= 315), 0.5, 0)  # West-facing
    
    # 4. Terrain roughness (sites often on relatively flat areas)
    roughness = np.std(elevation, axis=0) if elevation.ndim > 1 else 0
    if elevation.ndim > 1:
        roughness = np.std(elevation, axis=0)
        # Apply moving window for local roughness
        from scipy.ndimage import uniform_filter
        roughness = uniform_filter(roughness, size=3)
    else:
        roughness = np.zeros_like(elevation)
    
    return {
        'elevation_zones': elevation_zones,
        'slope_stability': slope_stability,
        'aspect_preference': aspect_preference,
        'terrain_roughness': roughness,
        'slope': slope,
        'aspect': aspect,
        'hillshade': hillshade,
        'curvature': curvature
    }


if __name__ == "__main__":
    # Example usage
    processor = TerrainProcessor()
    
    # Process a single tile
    tile_path = "asset/tile_0.tif"
    result = processor.process_tile(tile_path, save_intermediate=True)
    
    print(f"Processed tile shape: {result['feature_stack'].shape}")
    print(f"Feature channels: {result['feature_stack'].shape[0]}") 