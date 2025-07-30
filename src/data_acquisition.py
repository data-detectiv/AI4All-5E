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
import matplotlib
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAcquisition:
    """Handles data acquisition from Google Earth Engine"""
    
    def __init__(self, ee_credentials_path: Optional[str] = None):
        """
        Initialize Earth Engine connection
        
        Args:
            ee_credentials_path: Path to Earth Engine service account JSON file
        """
        try:
            if ee_credentials_path and os.path.exists(ee_credentials_path):
                # Use service account authentication
                credentials = ee.ServiceAccountCredentials(
                    'ai4all5e@ee-oppongfoster89.iam.gserviceaccount.com',
                    ee_credentials_path
                )
                ee.Initialize(credentials, project='ee-oppongfoster89')
                logger.info(f"Earth Engine initialized with service account: {ee_credentials_path}")
            else:
                # Try default authentication
                ee.Initialize(project='ee-oppongfoster89')
                logger.info("Earth Engine initialized with default authentication")
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
            ee.Filter.eq('ECO_NAME', 'Southwest Amazon moist forests')
        )

        # size = amazon.size().getInfo()
        # logger.info(f"Amazon region feature count: {size}")
        # if size == 0:
        #     raise ValueError("Amazon region filter returned no features. Check the filter or your Earth Engine permissions.")
        
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
                   max_pixels: int = 1e6) -> ee.batch.Task:
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
    
    def monitor_tasks(self, tasks: List[ee.batch.Task], timeout_hours: int = 2) -> None:
        """
        Monitor export tasks and log their status
        
        Args:
            tasks: List of Earth Engine tasks to monitor
            timeout_hours: Maximum time to wait for tasks in hours
        """
        import time
        from datetime import datetime, timedelta
        
        start_time = datetime.now()
        timeout = timedelta(hours=timeout_hours)
        
        logger.info(f"Starting to monitor {len(tasks)} export tasks with {timeout_hours} hour timeout")
        
        while True:
            current_time = datetime.now()
            if current_time - start_time > timeout:
                logger.warning(f"Timeout reached after {timeout_hours} hours. Stopping task monitoring.")
                break
                
            all_completed = True
            failed_tasks = []
            
            for i, task in enumerate(tasks):
                try:
                    status = task.status()
                    state = status['state']
                    
                    if state == 'COMPLETED':
                        logger.info(f"Task {i} completed successfully")
                    elif state == 'FAILED':
                        error_msg = status.get('error_message', 'Unknown error')
                        logger.error(f"Task {i} failed: {error_msg}")
                        failed_tasks.append(i)
                    elif state == 'RUNNING':
                        logger.info(f"Task {i} is running...")
                        all_completed = False
                    elif state == 'READY':
                        logger.info(f"Task {i} is ready to run...")
                        all_completed = False
                    elif state == 'CANCEL_REQUESTED':
                        logger.warning(f"Task {i} cancellation requested")
                        all_completed = False
                    elif state == 'CANCELLED':
                        logger.warning(f"Task {i} was cancelled")
                    else:
                        logger.info(f"Task {i} status: {state}")
                        all_completed = False
                        
                except Exception as e:
                    logger.error(f"Error checking status of task {i}: {e}")
                    all_completed = False
            
            if all_completed:
                logger.info("All export tasks completed")
                break
            
            if failed_tasks:
                logger.warning(f"Failed tasks: {failed_tasks}")
                # Continue monitoring other tasks
            
            time.sleep(60)  # Check every minute instead of 30 seconds
        
        # Final status report
        logger.info("Final task status report:")
        for i, task in enumerate(tasks):
            try:
                status = task.status()
                logger.info(f"Task {i}: {status['state']}")
            except Exception as e:
                logger.error(f"Could not get final status for task {i}: {e}")


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
    da = DataAcquisition("src/ee-oppongfoster89-eb504ec856ea.json")
    
    # Get Amazon region
    amazon_region = da.get_amazon_region()
    
    # Export tiles
    tasks = da.batch_export_tiles(amazon_region, "data/tiles")
    
    # Monitor tasks
    da.monitor_tasks(tasks) 