#!/usr/bin/env python3
"""
Simple Earth Engine test to diagnose the issue
"""

import ee
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ee_connection():
    """Test basic Earth Engine connection"""
    try:
        # Initialize with service account
        credentials = ee.ServiceAccountCredentials(
            'ai4all5e@ee-oppongfoster89.iam.gserviceaccount.com',
            'src/ee-oppongfoster89-3ab1d8063f07.json'
        )
        ee.Initialize(credentials)
        logger.info("✅ Earth Engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Earth Engine initialization failed: {e}")
        return False

def test_srtm_access():
    """Test SRTM data access"""
    try:
        # Try to access a small SRTM region
        srtm = ee.Image("USGS/SRTMGL1_003").select('elevation')
        
        # Small test region (1 degree x 1 degree)
        test_region = ee.Geometry.Rectangle([-60, -10, -59, -9])
        
        # Clip to small region
        srtm_clipped = srtm.clip(test_region)
        
        # Get basic info
        info = srtm_clipped.getInfo()
        logger.info("✅ SRTM data access successful")
        logger.info(f"   Image info: {info.get('type', 'Unknown')}")
        return True
    except Exception as e:
        logger.error(f"❌ SRTM data access failed: {e}")
        return False

def test_small_export():
    """Test a small export"""
    try:
        # Small test region
        test_region = ee.Geometry.Rectangle([-60, -10, -59, -9])
        srtm = ee.Image("USGS/SRTMGL1_003").select('elevation').clip(test_region)
        
        # Create export task
        task = ee.batch.Export.image.toDrive(
            image=srtm,
            region=test_region,
            scale=90,
            maxPixels=1e6,  # Small limit
            description='test_export',
            fileFormat='GeoTIFF'
        )
        
        task.start()
        logger.info("✅ Small export task started")
        logger.info(f"   Task ID: {task.id}")
        return task
    except Exception as e:
        logger.error(f"❌ Small export failed: {e}")
        return None

if __name__ == "__main__":
    logger.info("=== Earth Engine Diagnostic Test ===")
    
    # Test 1: Connection
    if not test_ee_connection():
        exit(1)
    
    # Test 2: Data access
    if not test_srtm_access():
        exit(1)
    
    # Test 3: Small export
    task = test_small_export()
    if task:
        logger.info("✅ All tests passed! The issue might be with large exports or region size.")
    else:
        logger.error("❌ Export test failed") 