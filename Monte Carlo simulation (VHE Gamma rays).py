# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:34:10 2024

@author: oranc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Defines random points
def random_points(N, center, radius, seed=42):
    np.random.seed(seed)
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, radius ** 2, N))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack((x, y))

# Generate random points with 2D Gaussian distribution

def gaussian_points(N, mean, cov, seed=42):
    return np.random.multivariate_normal(mean, cov, N)

# Calculates significance at a given resolution
def calculate_significance_at_pixels(points, N_on_radius, N_off_radius, center, resolution):
    # Create KD-tree for spatial indexing
    tree = cKDTree(points)

    # Initialize significance grid
    significance_grid = np.zeros(resolution)

    # Iterate over each pixel in the search_region
    x_grid, y_grid = np.meshgrid(np.linspace(center[0]-0.16, center[0]+0.16, resolution[0]),
                                  np.linspace(center[1]-0.16, center[1]+0.16, resolution[1]))

    for i in range(resolution[0]):
        for j in range(resolution[1]):
            # Query points within the current pixel
            idx_N_on = tree.query_ball_point([x_grid[i, j], y_grid[i, j]], N_on_radius)
            idx_N_off = tree.query_ball_point([x_grid[i, j], y_grid[i, j]], N_off_radius)

            # Count points within radii
            N_on_count = len(idx_N_on)
            N_off_count = len(idx_N_off)

            # Calculate significance
            alpha = 1 / 6 
            total_N_off_count = len(idx_N_off)
            S = (N_on_count - (alpha) * (total_N_off_count)) / np.sqrt(N_on_count + (alpha ** 2) * (total_N_off_count))
            
            # Assign significance to the corresponding grid cell
            significance_grid[i, j] = S

    return significance_grid

def main():
    np.random.seed(None)  

    # Parameters
    radius = 0.6
    center_on = (0.5, 0)
    center_off = (-0.5, 0)  # Center for N_off counting
    
    num_points = 850  #850
    
    # Parameters for N-on and N-off circles
    N_on_radius = 0.089
    N_off_radius = 0.089

    # Generate random points
    points_on = random_points(num_points, center_on, radius)
    points_off = random_points(num_points, center_off, radius)

    # Generate additional random points with 2D Gaussian distribution for N_on count
    num_injected_points = 250  #250
    mean_injected = center_on
    cov_injected = [[0.01, 0], [0, 0.01]]  # Covariance matrix for Gaussian distribution
    injected_points = gaussian_points(num_injected_points, mean_injected, cov_injected)

    # Calculate significance at every pixel inside the search_region
    resolution = (int(17), int(17))

    # Calculate significance for N_on points
    significance_grid_on = calculate_significance_at_pixels(np.vstack((points_on, injected_points)), N_on_radius, N_off_radius, center_on, resolution)
    
    # Calculate significance for N_off points
    significance_grid_off = calculate_significance_at_pixels(points_off, N_on_radius, N_off_radius, center_off, resolution)

    # Subtract the significance grid obtained from N_off points from that obtained from N_on points
    significance_grid = significance_grid_on - significance_grid_off

    # Find peak significance value
    peak_significance = np.max(significance_grid)

   
    # Print peak significance value and its coordinates
    print("Peak significance value:", peak_significance)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(significance_grid, extent=[center_on[0]-0.16, center_on[0]+0.16, center_on[1]-0.16, center_on[1]+0.16], cmap='jet')
    plt.colorbar(label='Significance')
    plt.xlabel('R.A (deg)')
    plt.ylabel('Declination (deg)')
  
    plt.title('Significance at every pixel inside the search region')
    plt.show()
    
    num_simulations = 1
    # Accumulate results across simulations
    total_significances_over_5_sigma = 0

    # For each simulation
    for _ in range(num_simulations):
        # Simulate N_on and N_off circles for each simulation
        points_simulation = random_points(num_points, center_on, radius)
        significance_values = calculate_significance_at_pixels(points_simulation, N_on_radius, N_off_radius, center_on, resolution)
        num_sigmas = len(np.where(significance_values >= 5)[0])  # Count values over 5 sigma
        total_significances_over_5_sigma += num_sigmas

    # Print total number of significances over 5 sigma
    print("Total number of significances over 5 sigma:", total_significances_over_5_sigma)

if __name__ == "__main__":
    main()
