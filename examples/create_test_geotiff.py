# File: examples/create_test_geotiff.py
"""
A simple script to generate a sample GeoTIFF file of a wrapped interferogram.
This file will be used as input for testing the command-line tool.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_sample_geotiff(output_path="sample_wrapped.tif", size=50, noise_level=0.2):
    """Generates and saves a noisy, wrapped phase image as a GeoTIFF."""
    print(f"Creating a {size}x{size} sample GeoTIFF with noise={noise_level}...")
    
    # Create synthetic data
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    ground_truth_phase = (x**2 - y**2) * 0.8
    noise = np.random.normal(0, noise_level, size=(size, size))
    wrapped_phase = np.angle(np.exp(1j * (ground_truth_phase + noise)))
    
    # Define GeoTIFF metadata
    transform = from_origin(0, 0, 1, 1) # Dummy transform
    profile = {
        'driver': 'GTiff',
        'height': size,
        'width': size,
        'count': 1,
        'dtype': wrapped_phase.dtype,
        'crs': None, # No coordinate reference system
        'transform': transform,
    }
    
    # Write to file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(wrapped_phase.astype(rasterio.float32), 1)
        
    print(f"Sample file saved to: {output_path}")

if __name__ == '__main__':
    create_sample_geotiff()
