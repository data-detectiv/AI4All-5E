"""
Main Pipeline for Archaeological Site Detection
Orchestrates the complete workflow from data acquisition to model deployment
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any
import numpy as np
import json
import ee
import shutil

# Add src to path
sys.path.append('src')

from data_acquisition import DataAcquisition, load_tile_data
from preprocessing import TerrainProcessor, create_archaeological_features
from annotation import ArchaeologicalFeatureDetector
from models import ModelTrainer, TerrainDataset, evaluate_model_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArchaeologicalSiteDetectionPipeline:
    """Complete pipeline for archaeological site detection"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.data_acquisition = None
        self.terrain_processor = None
        self.feature_detector = None
        self.model_trainer = None
        
        # Create output directories
        self._create_directories()
        
        logger.info("Initialized Archaeological Site Detection Pipeline")
    
    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            'data/tiles',
            'data/processed',
            'analysis',
            'models',
            'results',
            'visualizations'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def step1_data_acquisition(self):
        """Step 1: Data Acquisition from Google Earth Engine (API only, in-memory)"""
        logger.info("=== Step 1: Data Acquisition ===")
        # Initialize DataAcquisition
        if not self.data_acquisition:
            self.data_acquisition = DataAcquisition(self.config.get('ee_credentials_path'))
        # Get Amazon region
        amazon_region = self.data_acquisition.get_amazon_region()
        # Get SRTM data (full region)
        srtm_image = self.data_acquisition.get_srtm_data(amazon_region)
        # Create tiling grid
        num_tiles_x = self.config.get('num_tiles_x', 3)
        num_tiles_y = self.config.get('num_tiles_y', 3)
        tiles = self.data_acquisition.create_tiling_grid(amazon_region, num_tiles_x, num_tiles_y)
        # Download each tile as a numpy array (in-memory)
        tile_arrays = []
        for i, tile_geom in enumerate(tiles):
            # Get bounds
            coords = tile_geom.bounds().getInfo()['coordinates'][0]
            min_lon, min_lat = coords[0]
            max_lon, max_lat = coords[2]
            # Download tile as numpy array
            url = srtm_image.clip(tile_geom).getDownloadURL({
                'scale': 90,
                'crs': 'EPSG:4326',
                'region': tile_geom,
                'format': 'NPY'
            })
            import requests
            import io
            response = requests.get(url)
            elevation = np.load(io.BytesIO(response.content))
            metadata = {
                'bounds': (min_lon, min_lat, max_lon, max_lat),
                'crs': 'EPSG:4326',
                'nodata': None
            }
            tile_arrays.append({'elevation': elevation, 'metadata': metadata})
            logger.info(f"Fetched tile {i} from GEE API")
        logger.info(f"Data acquisition completed. {len(tile_arrays)} tiles fetched from GEE API.")
        return tile_arrays

    def step2_preprocessing(self, tile_arrays: list) -> list:
        """
        Step 2: Preprocess tiles and compute terrain derivatives (in-memory)
        Args:
            tile_arrays: List of dicts with 'elevation' and 'metadata'
        Returns:
            List of processed tile data
        """
        logger.info("=== Step 2: Preprocessing ===")
        self.terrain_processor = TerrainProcessor(
            target_size=self.config.get('target_size', (100, 100))
        )
        processed_tiles = []
        for i, tile in enumerate(tile_arrays):
            logger.info(f"Processing tile {i}")
            result = self.terrain_processor.process_tile_array(
                elevation=tile['elevation'],
                metadata=tile['metadata']
            )
            processed_tiles.append(result)
        logger.info(f"Preprocessing completed. {len(processed_tiles)} tiles processed.")
        return processed_tiles
    
    def step3_feature_detection(self, processed_tiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: Detect archaeological features using computer vision
        
        Args:
            processed_tiles: List of processed tile data
            
        Returns:
            List of analysis results
        """
        logger.info("=== Step 3: Feature Detection ===")
        
        try:
            # Initialize feature detector
            self.feature_detector = ArchaeologicalFeatureDetector(
                min_feature_size=self.config.get('min_feature_size', 50),
                max_feature_size=self.config.get('max_feature_size', 1000)
            )
            
            # Analyze tiles
            analyses = self.feature_detector.batch_analyze_tiles(processed_tiles, output_dir="analysis")
            
            # Create analysis report
            report_path = self.feature_detector.create_analysis_report(
                analyses=analyses,
                output_path=self.config.get('analysis_report_path', 'analysis/report.html')
            )
            
            logger.info(f"Feature detection completed. Analysis report: {report_path}")
            return analyses
            
        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            raise
    
    def step4_model_training(self, processed_tiles: List[Dict[str, Any]], 
                           analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 4: Train machine learning models
        
        Args:
            processed_tiles: List of processed tile data
            analyses: List of analysis results
            
        Returns:
            Training results
        """
        logger.info("=== Step 4: Model Training ===")
        
        try:
            # Prepare training data
            features = []
            labels = []
            
            for tile, analysis in zip(processed_tiles, analyses):
                # Use feature stack as input
                features.append(tile['feature_stack'])
                
                # Create label based on analysis
                assessment = analysis.get('overall_assessment', 'natural')
                if assessment == 'high_archaeological_potential':
                    label = 1
                elif assessment == 'moderate_archaeological_potential':
                    label = 1  # Treat moderate as positive
                else:
                    label = 0
                
                labels.append(label)
            
            features = np.array(features)
            labels = np.array(labels)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Create datasets
            train_dataset = TerrainDataset(X_train, y_train)
            test_dataset = TerrainDataset(X_test, y_test)
            
            # Create data loaders
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            # Train models
            results = {}
            
            for model_type in ['cnn', 'autoencoder']:
                logger.info(f"Training {model_type} model...")
                
                # Initialize trainer
                self.model_trainer = ModelTrainer(model_type=model_type)
                self.model_trainer.create_model(
                    input_channels=features.shape[1],
                    num_classes=2 if model_type == 'cnn' else None
                )
                self.model_trainer.setup_training(
                    learning_rate=self.config.get('learning_rate', 0.001)
                )
                
                # Train model
                history = self.model_trainer.train(
                    train_loader=train_loader,
                    val_loader=test_loader,
                    epochs=self.config.get('epochs', 50),
                    patience=self.config.get('patience', 10)
                )
                
                # Evaluate model
                predictions = self.model_trainer.predict(test_loader)
                
                if model_type == 'cnn':
                    # For CNN, convert probabilities to binary predictions
                    y_pred = (predictions > 0.5).astype(int)
                    metrics = evaluate_model_performance(y_test, y_pred, predictions)
                else:
                    # For autoencoder, use reconstruction error as anomaly score
                    # Higher error = more likely to be archaeological
                    threshold = np.percentile(predictions, 75)  # Use 75th percentile as threshold
                    y_pred = (predictions > threshold).astype(int)
                    metrics = evaluate_model_performance(y_test, y_pred, predictions)
                
                # Save model
                model_path = f"models/{model_type}_model.pth"
                self.model_trainer.save_model(model_path)
                
                results[model_type] = {
                    'history': history,
                    'metrics': metrics,
                    'model_path': model_path
                }
            
            # Save training results
            results_path = os.path.join('results', 'training_results.json')
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for model_type, result in results.items():
                    json_results[model_type] = {
                        'metrics': result['metrics'],
                        'model_path': result['model_path']
                    }
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Model training completed. Results saved to: {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def step5_deployment_preparation(self, analyses: List[Dict[str, Any]], 
                                   training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 5: Prepare for deployment and create final report
        
        Args:
            analyses: List of analysis results
            training_results: Training results
            
        Returns:
            Deployment summary
        """
        logger.info("=== Step 5: Deployment Preparation ===")
        
        try:
            # Create deployment summary
            deployment_summary = {
                'total_tiles_analyzed': len(analyses),
                'tiles_with_features': sum(1 for a in analyses if a.get('total_features', 0) > 0),
                'high_potential_tiles': sum(1 for a in analyses 
                                          if a.get('overall_assessment') == 'high_archaeological_potential'),
                'model_performance': training_results,
                'recommendations': self._generate_recommendations(analyses)
            }
            
            # Save deployment summary
            summary_path = os.path.join('results', 'deployment_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(deployment_summary, f, indent=2)
            
            # Create final visualization
            self._create_final_visualization(analyses)
            
            logger.info(f"Deployment preparation completed. Summary: {summary_path}")
            return deployment_summary
            
        except Exception as e:
            logger.error(f"Deployment preparation failed: {e}")
            raise
    
    def _generate_recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        high_potential_count = sum(1 for a in analyses 
                                 if a.get('overall_assessment') == 'high_archaeological_potential')
        
        if high_potential_count > 0:
            recommendations.append(f"Found {high_potential_count} tiles with high archaeological potential")
            recommendations.append("Recommend ground truthing and field surveys for these areas")
        
        total_features = sum(a.get('total_features', 0) for a in analyses)
        if total_features > 0:
            recommendations.append(f"Detected {total_features} potential archaeological features")
            recommendations.append("Consider using higher resolution data for detailed analysis")
        
        recommendations.append("Implement bias mitigation strategies for canopy and cloud coverage")
        recommendations.append("Engage with local communities and indigenous knowledge holders")
        
        return recommendations
    
    def _create_final_visualization(self, analyses: List[Dict[str, Any]]):
        """Create final summary visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature type distribution
        feature_types = {}
        for analysis in analyses:
            for feature_type, count in analysis.get('feature_types', {}).items():
                feature_types[feature_type] = feature_types.get(feature_type, 0) + count
        
        if feature_types:
            axes[0, 0].pie(feature_types.values(), labels=feature_types.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Feature Type Distribution')
        
        # Confidence distribution
        confidences = [a.get('average_confidence', 0) for a in analyses]
        axes[0, 1].hist(confidences, bins=10, alpha=0.7)
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Number of Tiles')
        
        # Assessment distribution
        assessments = [a.get('overall_assessment', 'unknown') for a in analyses]
        assessment_counts = {}
        for assessment in assessments:
            assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        
        axes[1, 0].bar(assessment_counts.keys(), assessment_counts.values())
        axes[1, 0].set_title('Assessment Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Feature count vs confidence
        feature_counts = [a.get('total_features', 0) for a in analyses]
        axes[1, 1].scatter(feature_counts, confidences, alpha=0.6)
        axes[1, 1].set_title('Feature Count vs Confidence')
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('Confidence Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', 'final_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Returns:
            Pipeline results
        """
        logger.info("Starting Archaeological Site Detection Pipeline")
        
        try:
            # Step 1: Data Acquisition
            tile_arrays = self.step1_data_acquisition()
            
            # Step 2: Preprocessing
            processed_tiles = self.step2_preprocessing(tile_arrays)
            
            # Step 3: Feature Detection
            analyses = self.step3_feature_detection(processed_tiles)
            
            # Step 4: Model Training
            training_results = self.step4_model_training(processed_tiles, analyses)
            
            # Step 5: Deployment Preparation
            deployment_summary = self.step5_deployment_preparation(analyses, training_results)
            
            logger.info("Pipeline completed successfully!")
            
            return {
                'tile_arrays': tile_arrays,
                'processed_tiles': len(processed_tiles),
                'analyses': analyses,
                'training_results': training_results,
                'deployment_summary': deployment_summary
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Archaeological Site Detection Pipeline')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--step', type=str, choices=['all', '1', '2', '3', '4', '5'],
                       default='all', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration with service account path
        config = {
            'ee_credentials_path': 'src/ee-oppongfoster89-3ab1d8063f07.json',
            'buffer_degrees': 0.1,
            'num_tiles_x': 5,
            'num_tiles_y': 5,
            'target_size': [100, 100],
            'min_feature_size': 50,
            'max_feature_size': 1000,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10,
            'tile_output_dir': 'data/tiles',
            'analysis_output_dir': 'analysis',
            'analysis_report_path': 'analysis/report.html'
        }
    
    # Initialize pipeline
    pipeline = ArchaeologicalSiteDetectionPipeline(config)
    
    if args.step == 'all':
        # Run complete pipeline
        results = pipeline.run_pipeline()
        print("Pipeline completed successfully!")
        print(f"Results: {json.dumps(results, indent=2)}")
    else:
        # Run specific step
        print(f"Running step {args.step} only")
        # Implementation for running specific steps would go here


if __name__ == "__main__":
    main() 