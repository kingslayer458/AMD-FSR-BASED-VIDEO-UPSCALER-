import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

class FSRVideoUpscaler:
    def __init__(self):
        """Initialize the FSR-inspired video upscaler."""
        self.threshold_resolution = (720, 480)  # Example threshold for SD videos
        
    def should_upscale(self, video_path):
        """Check if the video resolution is below the threshold."""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return width < self.threshold_resolution[0] or height < self.threshold_resolution[1]
    
    def _edge_adaptive_spatial_upscaling(self, frame, target_size):
        """Apply edge-adaptive spatial upscaling (inspired by FSR's EASU step)."""
        # Step 1: Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Step 3: Dilate the edges slightly to enhance edge regions
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Step 4: Create an edge mask
        edge_mask = edges_dilated > 0
        
        # Step 5: Upscale using different methods
        # For edge areas - use a sharper upscaling
        upscaled_sharp = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # For non-edge areas - use smoother upscaling
        upscaled_smooth = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Step 6: Create the output frame
        result = np.zeros(upscaled_sharp.shape, dtype=np.uint8)
        
        # Expand edge mask to match the upscaled dimensions
        edge_mask_upscaled = cv2.resize(edge_mask.astype(np.uint8), target_size, 
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Apply edge mask to combine both upscaling results
        for c in range(3):  # For each color channel
            result[:, :, c] = np.where(edge_mask_upscaled, 
                                       upscaled_sharp[:, :, c], 
                                       upscaled_smooth[:, :, c])
        
        return result
    
    def _robust_contrast_adaptive_sharpening(self, frame):
        """Apply contrast adaptive sharpening (inspired by FSR's RCAS step)."""
        # Apply unsharp mask filter with adaptive amount
        blur = cv2.GaussianBlur(frame, (0, 0), 3)
        
        # Calculate local variance to determine sharpening strength
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        local_var = cv2.GaussianBlur(gray * gray, (7, 7), 0) - \
                   cv2.GaussianBlur(gray, (7, 7), 0) ** 2
        
        # Normalize variance to range 0-1
        strength_mask = np.clip(local_var / (local_var.max() + 0.0001), 0, 1)
        
        # Apply adaptive sharpening
        sharpened = frame + 0.8 * (frame - blur)
        
        # Blend based on local variance
        result = np.zeros_like(frame)
        for c in range(3):
            strength_3d = np.expand_dims(strength_mask, axis=2).repeat(3, axis=2)[:, :, c]
            result[:, :, c] = frame[:, :, c] + strength_3d * (sharpened[:, :, c] - frame[:, :, c])
        
        # Clip values to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def upscale_frame(self, frame, target_size):
        """Upscale a single frame using FSR-inspired approach."""
        # Step 1: Edge Adaptive Spatial Upscaling (EASU)
        upscaled = self._edge_adaptive_spatial_upscaling(frame, target_size)
        
        # Step 2: Robust Contrast Adaptive Sharpening (RCAS)
        enhanced = self._robust_contrast_adaptive_sharpening(upscaled)
        
        return enhanced
    
    def _process_chunk(self, args):
        """Process a chunk of frames for parallel processing."""
        input_path, output_path, start_frame, end_frame, width, height, fps = args
        
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create a temporary output file for this chunk
        temp_output = f"{output_path}_chunk_{start_frame}_{end_frame}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width * 2, height * 2))
        
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Upscale the frame
            upscaled_frame = self.upscale_frame(frame, (width * 2, height * 2))
            
            # Write to output video
            out.write(upscaled_frame)
        
        cap.release()
        out.release()
        
        return temp_output
        
    def upscale_video(self, input_path, output_path, parallel=True):
        """Upscale an entire video file with option for parallel processing."""
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if not parallel:
            # Process sequentially
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2))
            
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Upscale the frame
                upscaled_frame = self.upscale_frame(frame, (width * 2, height * 2))
                
                # Write to output video
                out.write(upscaled_frame)
                
                # Update progress
                frame_number += 1
                progress = (frame_number / frame_count) * 100
                print(f"Upscaling progress: {progress:.2f}%", end='\r')
            
            cap.release()
            out.release()
        else:
            # Process in parallel using multiple CPU cores
            num_cores = max(1, cpu_count() - 1)  # Leave one core free
            chunk_size = frame_count // num_cores
            
            chunks = []
            for i in range(num_cores):
                start_frame = i * chunk_size
                end_frame = start_frame + chunk_size if i < num_cores - 1 else frame_count
                chunks.append((input_path, output_path, start_frame, end_frame, width, height, fps))
            
            # Process chunks in parallel
            print(f"Processing video in {num_cores} parallel chunks...")
            with Pool(num_cores) as pool:
                temp_files = pool.map(self._process_chunk, chunks)
            
            # Combine the temporary files
            self._combine_video_chunks(temp_files, output_path)
            
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print("\nVideo upscaling completed!")
        return output_path
    
    def _combine_video_chunks(self, chunk_files, output_path):
        """Combine multiple video chunks into a single output file."""
        # Create a file listing for ffmpeg
        list_file = "filelist.txt"
        with open(list_file, "w") as f:
            for chunk_file in chunk_files:
                f.write(f"file '{chunk_file}'\n")
        
        # Use ffmpeg to concatenate the files
        os.system(f"ffmpeg -f concat -safe 0 -i {list_file} -c copy {output_path}")
        
        # Clean up
        os.remove(list_file)

# Integration with your streaming platform
def process_uploaded_video(video_path, output_dir):
    """Process a newly uploaded video and upscale if needed."""
    # Initialize the FSR upscaler
    upscaler = FSRVideoUpscaler()
    
    # Check if the video needs upscaling
    if upscaler.should_upscale(video_path):
        # Generate output path
        filename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"fsr_upscaled_{filename}")
        
        # Perform upscaling
        upscaled_video = upscaler.upscale_video(video_path, output_path, parallel=True)
        
        # Return the path to the upscaled video
        return upscaled_video
    
    # If no upscaling needed, return the original path
    return video_path

# Example usage in your application
if __name__ == "__main__":
    # This could be triggered by your video upload handler
    uploaded_video = "/path/to/uploaded/video.mp4"
    output_directory = "/path/to/processed/videos"
    
    # Process the video
    final_video_path = process_uploaded_video(uploaded_video, output_directory)
    
    # Update the video reference in your database
    print(f"Video processing complete. Final video at: {final_video_path}")