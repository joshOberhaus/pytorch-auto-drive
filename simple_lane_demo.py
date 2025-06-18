#!/usr/bin/env python3
"""
Simple Lane Detection Visualization Example

This script demonstrates the basic workflow:
1. Download test images
2. Download a pre-trained model  
3. Run visualization
4. Display results

Usage: python simple_lane_demo.py
"""

import os
import subprocess
import urllib.request
import zipfile

def download_google_drive_file(file_id, output_path):
    """Download file from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    print(f"Downloading {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def main():
    print("=== Simple Lane Detection Demo ===")
    
    # Check if we're in the right directory
    if not os.path.exists('main_landet.py'):
        print("Error: Please run this script from the pytorch-auto-drive root directory")
        return
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('demo_results', exist_ok=True)
    
    # 1. Download test images (129MB)
    test_images_path = 'PAD_test_images'
    if not os.path.exists(test_images_path):
        print("\n1. Downloading test images...")
        zip_file = 'PAD_test_images.zip'
        
        if download_google_drive_file('1XQvBS1uoHeIgUv7oDQ4Vp1tWYi0oAGhU', zip_file):
            print("Extracting test images...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zip_file)
            print("Test images ready!")
        else:
            print("Failed to download test images")
            return
    else:
        print("1. Test images already available")
    
    # 2. Download ERFNet model (recommended for demo)
    model_path = 'checkpoints/erfnet_baseline_culane_20210204.pt'
    if not os.path.exists(model_path):
        print("\n2. Downloading ERFNet model...")
        if not download_google_drive_file('16-Q_jZYc9IIKUEHhClSTwZI4ClMeVvQS', model_path):
            print("Failed to download model")
            return
    else:
        print("2. Model already available")
    # TODO: still need to download pretrained weights for erfnet

    # 3. Run visualization on CULane test images
    print("\n3. Running lane detection visualization...")
    
    image_dir = os.path.join(test_images_path, 'lane_test_images', '05171008_0748.MP4')
    output_dir = 'demo_results/culane_visualization'
    config_file = 'configs/lane_detection/baseline/erfnet_culane.py'
    
    # Create visualization command
    cmd = [
        'python', 'tools/vis/lane_img_dir.py',
        f'--image-path={image_dir}',
        '--image-suffix=.jpg',
        f'--save-path={output_dir}',
        '--pred',
        f'--config={config_file}',
        f'--checkpoint={model_path}',
        '--mixed-precision'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Visualization completed successfully!")
            print(f"Results saved to: {output_dir}")
            
            # List generated files
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"Generated {len(files)} visualization files:")
                for f in files[:5]:  # Show first 5 files
                    print(f"  - {f}")
                if len(files) > 5:
                    print(f"  ... and {len(files)-5} more")
        else:
            print("❌ Visualization failed:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running visualization: {e}")
        return
    
    # 4. Optional: Run video visualization
    print("\n4. Running video visualization (optional)...")
    video_input = os.path.join(test_images_path, 'lane_test_images', 'tusimple_val_1min.avi')
    video_output = 'demo_results/tusimple_prediction.avi'
    
    if os.path.exists(video_input):
        video_cmd = [
            'python', 'tools/vis/lane_video.py',
            f'--video-path={video_input}',
            f'--save-path={video_output}',
            '--config=configs/lane_detection/baseline/erfnet_tusimple.py',
            f'--checkpoint=checkpoints/erfnet_baseline_tusimple_20210424.pt'
        ]
        
        # Download TuSimple model if needed
        tusimple_model = 'checkpoints/erfnet_baseline_tusimple_20210424.pt'
        if not os.path.exists(tusimple_model):
            print("Downloading TuSimple model...")
            download_google_drive_file('1rLWDP_dkIQ7sBsCEzJi8T7ET1EPghhJJ', tusimple_model)
        
        if os.path.exists(tusimple_model):
            print(f"Running: {' '.join(video_cmd)}")
            try:
                result = subprocess.run(video_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Video visualization completed: {video_output}")
                else:
                    print("Video visualization failed (this is optional)")
            except Exception as e:
                print(f"Video processing error: {e}")
    
    print("\n=== Demo completed! ===")
    print(f"Check the results in:")
    print(f"  - Images: {output_dir}")
    if os.path.exists(video_output):
        print(f"  - Video: {video_output}")
    
    print("\nNext steps:")
    print("1. Open the generated images to see lane detection results")
    print("2. Try different models with the full lane_detection_demo.py script")
    print("3. Run evaluation with: python main_landet.py --test --config=... --checkpoint=...")

if __name__ == '__main__':
    main()
