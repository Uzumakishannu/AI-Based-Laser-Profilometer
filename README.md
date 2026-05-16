# AI-Based Laser Profilometer

AI-Based Laser Profilometer is a real-time computer vision system for surface defect detection and 3D surface visualization using structured laser imaging. The system captures laser line distortions from object surfaces and processes them through OpenCV pipelines to analyze surface irregularities, reconstruct depth profiles, and generate interactive 3D visualizations.

## Overview

The project combines computer vision, real-time processing, and hardware integration to deliver a practical workflow for surface inspection and profilometry. It is designed for rapid experimentation and debugging while supporting structured light surface acquisition.

## Features

- Real-time laser line detection
- Surface defect identification
- 3D surface profile visualization
- Structured light image processing
- Real-time frame acquisition and analysis
- Depth-based surface reconstruction
- Modular OpenCV processing pipeline
- Optimized workflow for rapid experimentation and debugging

## Tech Stack

- Python
- OpenCV
- NumPy
- Jetson Nano
- Raspberry Pi Camera
- Structured Laser Imaging

## System Workflow

1. A laser line is projected onto the target surface.
2. The Raspberry Pi Camera captures the laser deformation pattern.
3. Frames are processed using OpenCV pipelines.
4. Laser distortions are extracted and analyzed.
5. Surface defects and irregularities are identified.
6. Processed depth information is used for 3D surface visualization.

## Hardware Components

- Jetson Nano
- Raspberry Pi Camera
- Laser Module
- DC Power Supply
- Mounting Frame

## Computer Vision Pipeline

The system uses multiple OpenCV-based processing stages, including:

- Frame acquisition
- Grayscale conversion
- Gaussian filtering
- Thresholding
- Laser line extraction
- Edge detection
- Surface contour analysis
- Depth map generation
- 3D visualization

## Applications

- Surface roughness analysis
- Industrial defect inspection
- Structural surface analysis
- Manufacturing quality inspection
- Non-contact surface measurement

## Project Goals

- Build a low-cost real-time profilometer system
- Explore structured light imaging techniques
- Develop optimized computer vision workflows
- Generate real-time 3D surface visualizations
- Improve defect analysis accuracy using image processing

## Future Improvements

- AI-based defect classification
- Faster GPU-accelerated processing
- Web dashboard for remote monitoring
- Advanced 3D reconstruction
- Automated calibration workflows
- Real-time defect severity analysis

## Usage

### Install requirements

```powershell
python -m pip install -r requirements.txt
```

### Run the main dashboard

```powershell
python profilometer_simple_grid_ui.py --input "your_video_path.mp4" --display --save-html
```

### Run with any video without editing code

```powershell
python profilometer_simple_grid_ui.py --input "C:\path\to\your_video.mp4" --save-html
```

### Run a folder of videos

```powershell
python profilometer_simple_grid_ui.py --input-dir "C:\path\to\video_folder" --output-dir "C:\path\to\analysis_outputs" --save-html
```

## Output

For each video, the program generates:

- an inspection dashboard video
- a JSON analysis report
- an optional interactive 3D surface HTML file

## Author

Suryavamsi Abhishek

B.Tech ECE — IIITDM Kancheepuram
