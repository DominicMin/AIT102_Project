
import cv2
import numpy as np

def create_dummy_video(filename="test_video.mp4", width=640, height=480, seconds=5, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    frames = seconds * fps
    print(f"Generating {filename} ({frames} frames)...")
    
    for i in range(frames):
        # Create a black image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw moving circles
        x1 = int(width/2 + (width/3) * np.sin(i * 0.1))
        y1 = int(height/2 + (height/3) * np.cos(i * 0.1))
        cv2.circle(img, (x1, y1), 50, (255, 0, 0), -1)
        
        x2 = int(width/2 + (width/3) * np.cos(i * 0.15))
        y2 = int(height/2 + (height/3) * np.sin(i * 0.15))
        cv2.circle(img, (x2, y2), 70, (0, 255, 0), -1)
        
        # Moving text
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(img)
        
    out.release()
    print("Video generated successfully.")

if __name__ == "__main__":
    create_dummy_video()
