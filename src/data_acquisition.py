"""
Data Acquisition Module for Archaeological Site Detection
Handles Google Earth Engine integration and SRTM data extraction
"""

import ee
import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAcquisition:
    """Handles data acquisition from Google Earth Engine"""
    
    def __init__(self, ee_credentials_path: Optional[str] = None):
        """
        Initialize Earth Engine connection
        
        Args:
            ee_credentials_path: Path to Earth Engine credentials file
        """
        try:
            if ee_credentials_path and os.path.exists(ee_credentials_path):
                ee.Authenticate()
            ee.Initialize()
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            raise
    
    def get_amazon_region(self, buffer_degrees: float = 0.1) -> ee.Geometry:
        """
        Get Amazon biome region with optional buffer
        
        Args:
            buffer_degrees: Buffer around Amazon biome in degrees
            
        Returns:
            Earth Engine geometry of Amazon region
        """
        # Load Amazon biome from RESOLVE ecoregions
        amazon = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017").filter(
            ee.Filter.eq('ECO_NAME', 'Amazonia')
        )
        
        # Add buffer if specified
        if buffer_degrees > 0:
            amazon = amazon.geometry().buffer(buffer_degrees)
        else:
            amazon = amazon.geometry()
            
        return amazon
    
    def get_srtm_data(self, region: ee.Geometry, dataset: str = "USGS/SRTMGL1_003") -> ee.Image:
        """
        Get SRTM elevation data for specified region
        
        Args:
            region: Earth Engine geometry defining the region
            dataset: SRTM dataset to use
            
        Returns:
            Earth Engine image of elevation data
        """
        srtm = ee.Image(dataset).select('elevation').clip(region)
        return srtm
    
    def create_tiling_grid(self, region: ee.Geometry, 
                          num_tiles_x: int = 5, 
                          num_tiles_y: int = 5) -> List[ee.Geometry]:
        """
        Create a grid of tiles for processing
        
        Args:
            region: Bounding box region
            num_tiles_x: Number of tiles in x direction
            num_tiles_y: Number of tiles in y direction
            
        Returns:
            List of tile geometries
        """
        # Get bounding box coordinates
        coords = region.bounds().coordinates().get(0)
        coords = ee.List(coords)
        
        min_lon = ee.Number(ee.List(coords.get(0)).get(0))
        min_lat = ee.Number(ee.List(coords.get(0)).get(1))
        max_lon = ee.Number(ee.List(coords.get(2)).get(0))
        max_lat = ee.Number(ee.List(coords.get(2)).get(1))
        
        # Calculate step sizes
        step_lon = max_lon.subtract(min_lon).divide(num_tiles_x)
        step_lat = max_lat.subtract(min_lat).divide(num_tiles_y)
        
        tiles = []
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                tile_min_lon = min_lon.add(step_lon.multiply(i))
                tile_min_lat = min_lat.add(step_lat.multiply(j))
                tile_max_lon = tile_min_lon.add(step_lon)
                tile_max_lat = tile_min_lat.add(step_lat)
                
                tile = ee.Geometry.BBox(
                    tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat
                )
                tiles.append(tile)
        
        return tiles
    
    def export_tile(self, image: ee.Image, tile: ee.Geometry, 
                   output_path: str, scale: int = 90, 
                   max_pixels: int = 1e8) -> ee.batch.Task:
        """
        Export a tile to Google Drive or Cloud Storage
        
        Args:
            image: Earth Engine image to export
            tile: Tile geometry
            output_path: Output file path
            scale: Resolution in meters
            max_pixels: Maximum pixels per export
            
        Returns:
            Earth Engine export task
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            region=tile,
            scale=scale,
            maxPixels=max_pixels,
            description=os.path.basename(output_path),
            fileFormat='GeoTIFF'
        )
        return task
    
    def batch_export_tiles(self, region: ee.Geometry, 
                          output_dir: str = "data/tiles",
                          num_tiles_x: int = 5, 
                          num_tiles_y: int = 5) -> List[ee.batch.Task]:
        """
        Export multiple tiles in batch
        
        Args:
            region: Region to tile
            output_dir: Output directory for tiles
            num_tiles_x: Number of tiles in x direction
            num_tiles_y: Number of tiles in y direction
            
        Returns:
            List of export tasks
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get SRTM data
        srtm = self.get_srtm_data(region)
        
        # Create tiling grid
        tiles = self.create_tiling_grid(region, num_tiles_x, num_tiles_y)
        
        # Export each tile
        tasks = []
        for i, tile in enumerate(tiles):
            output_path = os.path.join(output_dir, f"tile_{i}.tif")
            task = self.export_tile(srtm, tile, output_path)
            task.start()
            tasks.append(task)
            logger.info(f"Started export: tile_{i}.tif")
        
        return tasks
    
    def monitor_tasks(self, tasks: List[ee.batch.Task]) -> None:
        """
        Monitor export tasks and log their status
        
        Args:
            tasks: List of Earth Engine tasks to monitor
        """
        import time
        
        while True:
            all_completed = True
            for i, task in enumerate(tasks):
                status = task.status()
                state = status['state']
                
                if state == 'COMPLETED':
                    logger.info(f"Task {i} completed successfully")
                elif state == 'FAILED':
                    logger.error(f"Task {i} failed: {status.get('error_message', 'Unknown error')}")
                elif state == 'RUNNING':
                    logger.info(f"Task {i} is running...")
                    all_completed = False
                else:
                    logger.info(f"Task {i} status: {state}")
                    all_completed = False
            
            if all_completed:
                logger.info("All export tasks completed")
                break
            
            time.sleep(30)  # Check every 30 seconds


def load_tile_data(tile_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load tile data from GeoTIFF file
    
    Args:
        tile_path: Path to GeoTIFF file
        
    Returns:
        Tuple of (elevation_data, metadata)
    """
    with rasterio.open(tile_path) as src:
        elevation = src.read(1)
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'nodata': src.nodata,
            'width': src.width,
            'height': src.height
        }
        
        # Handle nodata values
        if src.nodata is not None:
            elevation = np.ma.masked_equal(elevation, src.nodata)
        
        return elevation, metadata


if __name__ == "__main__":
    # Example usage
    da = DataAcquisition()
    
    # Get Amazon region
    amazon_region = da.get_amazon_region()
    
    # Export tiles
    tasks = da.batch_export_tiles(amazon_region, "data/tiles")
    
    # Monitor tasks
    da.monitor_tasks(tasks) 