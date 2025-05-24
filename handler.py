import runpod
import os
import json
import requests
import cv2
import argparse # Keep for Namespace compatibility if useful
import glob
import torch
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from scipy.ndimage import gaussian_filter1d
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.video_util import VideoReader, VideoWriter
from basicsr.utils.registry import ARCH_REGISTRY
# from torch.hub import download_url_to_file # Already have load_file_from_url

# --- Global Model Initialization START ---
print("Starting global model initialization...")
device = get_device()
print(f"Using device: {device}")

# Configuration for models and paths (adapted from hugging_face/app.py)
MODEL_BASE_URL = "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/"
WEIGHTS_DIR = "/app/weights/" # Standard directory within the Docker image

# Ensure weights directory exists
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(os.path.join(WEIGHTS_DIR, "KEEP"), exist_ok=True)
os.makedirs(os.path.join(WEIGHTS_DIR, "facelib"), exist_ok=True)
os.makedirs(os.path.join(WEIGHTS_DIR, "realesrgan"), exist_ok=True)

# Load KEEP model
keep_model_config = {
    'architecture': {
        'img_size': 512, 'emb_dim': 256, 'dim_embd': 512, 'n_head': 8, 'n_layers': 9,
        'codebook_size': 1024, 'cft_list': ['16', '32', '64'], 'kalman_attn_head_dim': 48,
        'num_uncertainty_layers': 3, 'cfa_list': ['16', '32'], 'cfa_nhead': 4, 'cfa_dim': 256, 'cond': 1
    },
    'checkpoint_url': os.path.join(MODEL_BASE_URL, 'KEEP-b76feb75.pth'),
    'checkpoint_dir': os.path.join(WEIGHTS_DIR, "KEEP")
}
keep_net = ARCH_REGISTRY.get('KEEP')(**keep_model_config['architecture']).to(device)
keep_ckpt_path = load_file_from_url(url=keep_model_config['checkpoint_url'], model_dir=keep_model_config['checkpoint_dir'], progress=True, file_name=None)
keep_checkpoint = torch.load(keep_ckpt_path, map_location=device, weights_only=True) # Added map_location
keep_net.load_state_dict(keep_checkpoint['params_ema'])
keep_net.eval()
print("KEEP model loaded.")

# Function to set up RealESRGAN (from hugging_face/app.py)
def set_realesrgan_upsampler():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer
    
    use_half = False
    if torch.cuda.is_available(): # Check device availability properly
        if device.type == 'cuda':
            use_half = True 

    model_path = load_file_from_url(
        url=os.path.join(MODEL_BASE_URL, 'RealESRGAN_x2plus.pth'),
        model_dir=os.path.join(WEIGHTS_DIR, "realesrgan"), progress=True, file_name=None
    )
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(scale=2, model_path=model_path, model=model, tile=400, tile_pad=40, pre_pad=0, half=use_half, device=device) # Pass device
    if device.type == 'cpu':
        import warnings
        warnings.warn('RealESRGAN: Running on CPU now! Make sure your PyTorch version matches your CUDA. The unoptimized RealESRGAN is slow on CPU.', category=RuntimeWarning)
    print("RealESRGAN upsampler initialized.")
    return upsampler

FACELIB_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "facelib")
load_file_from_url(url=os.path.join(MODEL_BASE_URL, 'detection_Resnet50_Final.pth'), model_dir=FACELIB_WEIGHTS_DIR, progress=True, file_name=None)
load_file_from_url(url=os.path.join(MODEL_BASE_URL, 'detection_mobilenet0.25_Final.pth'), model_dir=FACELIB_WEIGHTS_DIR, progress=True, file_name=None)
load_file_from_url(url=os.path.join(MODEL_BASE_URL, 'yolov5n-face.pth'), model_dir=FACELIB_WEIGHTS_DIR, progress=True, file_name=None)
load_file_from_url(url=os.path.join(MODEL_BASE_URL, 'yolov5l-face.pth'), model_dir=FACELIB_WEIGHTS_DIR, progress=True, file_name=None)
load_file_from_url(url=os.path.join(MODEL_BASE_URL, 'parsing_parsenet.pth'), model_dir=FACELIB_WEIGHTS_DIR, progress=True, file_name=None)
print("Facelib helper models downloaded.")

def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)
    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))
        interpolated_sequence[missing_indices] = np.interp(x[missing_indices], x[valid_indices], sequence[valid_indices])
    return interpolated_sequence

print("Global model initialization complete.")
# --- Global Model Initialization END ---

def enhance_video_file(input_video_path, output_dir, job_input):
    args = argparse.Namespace(
        input_path=input_video_path, 
        upscale=1, 
        max_length=job_input.get('max_processing_length', 200), # Allow longer processing, default 20 from original, 200 for serverless
        has_aligned=job_input.get('has_aligned', False),
        only_center_face=job_input.get('only_center_face', True),
        draw_box=job_input.get('draw_box', False),
        detection_model=job_input.get('detection_model', 'retinaface_resnet50'),
        bg_enhancement=job_input.get('bg_enhancement', False),
        face_upsample=job_input.get('face_upsample', False),
        bg_tile=job_input.get('bg_tile', 400),
        suffix=None, 
        save_video_fps=job_input.get('save_video_fps', None),
        model_type='KEEP', 
    )

    os.makedirs(output_dir, exist_ok=True)

    bg_upsampler_instance = None # Renamed to avoid conflict with global function name
    if args.bg_enhancement:
        bg_upsampler_instance = set_realesrgan_upsampler() 
    
    face_upsampler_instance = None # Renamed
    if args.face_upsample:
        face_upsampler_instance = bg_upsampler_instance if bg_upsampler_instance is not None else set_realesrgan_upsampler()

    net = keep_net 
    
    print(f"Processing video: {args.input_path}")
    if not args.has_aligned:
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler_instance is not None:
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

    face_helper = FaceRestoreHelper(
        args.upscale, 
        face_size=512, 
        crop_ratio=(1, 1), 
        det_model=args.detection_model, 
        save_ext='png', 
        use_parse=True, 
        device=device,
        model_rootpath=FACELIB_WEIGHTS_DIR 
    )

    input_img_list = []
    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input video file not found at: {args.input_path}")
        
    if args.input_path.lower().endswith(('mp4', 'mov', 'avi')):
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps
        vidreader.close()
        clip_name = os.path.splitext(os.path.basename(args.input_path))[0]
    else:
        raise TypeError(f'Unrecognized type of input video {args.input_path}.')
    
    if len(input_img_list) == 0:
        raise ValueError('No frames found in the input video.')

    print('Detecting keypoints and smooth alignment ...')
    if not args.has_aligned:
        raw_landmarks = []
        for i, img in enumerate(input_img_list):
            face_helper.clean_all()
            face_helper.read_image(img)
            num_det_faces = face_helper.get_face_landmarks_5(only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5, only_keep_largest=True)
            if num_det_faces == 1:
                raw_landmarks.append(face_helper.all_landmarks_5[0].reshape((10,)))
            elif num_det_faces == 0:
                raw_landmarks.append(np.array([np.nan]*10))
            else: # Multiple faces detected, pick the first one for simplicity in serverless
                print(f"Warning: Multiple faces ({num_det_faces}) detected in frame {i}. Using the first one.")
                raw_landmarks.append(face_helper.all_landmarks_5[0].reshape((10,)))

        raw_landmarks = np.array(raw_landmarks)
        for i_lm in range(raw_landmarks.shape[1]): # Iterate over landmark coordinates
            raw_landmarks[:, i_lm] = interpolate_sequence(raw_landmarks[:, i_lm]) 
        video_length = len(input_img_list)
        avg_landmarks = gaussian_filter1d(raw_landmarks, 5, axis=0).reshape(video_length, 5, 2)
    
    cropped_faces = []
    for i, img in enumerate(input_img_list):
        face_helper.clean_all()
        face_helper.read_image(img)
        if not args.has_aligned: # Use avg_landmarks if alignment was performed
             face_helper.all_landmarks_5 = [avg_landmarks[i]]
        # If args.has_aligned is true, the original app.py assumes faces are already cropped and aligned.
        # This part of logic might need more complex handling if has_aligned is true for serverless.
        # For now, proceed assuming align_warp_face is always needed if not pre-aligned.
        face_helper.align_warp_face() 
        if not face_helper.cropped_faces:
            # Fallback: if no face is cropped (e.g. alignment failed), skip frame or use placeholder
            # For now, creating a dummy black image tensor to avoid crash
            print(f"Warning: No face cropped for frame {i}. Using a black placeholder.")
            cropped_faces.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=device) * -1) # Normalized to -1 to 1 range
            continue

        cropped_face_t = img2tensor(face_helper.cropped_faces[0] / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_faces.append(cropped_face_t)
        
    if not cropped_faces: # If all frames failed cropping
        raise ValueError("Could not crop any faces from the video.")

    cropped_faces = torch.stack(cropped_faces, dim=0).unsqueeze(0).to(device)
    
    print('Restoring faces ...')
    with torch.no_grad():
        video_length = cropped_faces.shape[1]
        output_frames_list = [] 
        for start_idx in range(0, video_length, args.max_length):
            end_idx = min(start_idx + args.max_length, video_length)
            current_segment_input = cropped_faces[:, start_idx:end_idx, ...]
            if current_segment_input.shape[1] == 0: # Skip if segment is empty
                continue
            if current_segment_input.shape[1] == 1 and start_idx+1 < video_length : # Original bug: if end_idx-start_idx == 1
                 # Duplicate frame for sequence if it's a single frame segment AND not the very last frame of video
                output_frames_list.append(net(current_segment_input.repeat(1,2,1,1,1), need_upscale=False)[:, 0:1, ...])
            elif current_segment_input.shape[1] == 1 and start_idx+1 >= video_length: # Very last frame is single
                output_frames_list.append(net(current_segment_input, need_upscale=False))
            else:
                output_frames_list.append(net(current_segment_input, need_upscale=False))
        
        if not output_frames_list:
             raise ValueError("No output frames generated by the KEEP model.")

        output_tensor = torch.cat(output_frames_list, dim=1).squeeze(0) 
        # Expected: output_tensor.shape[0] == video_length (number of cropped faces)
        # Actual: output_tensor.shape[0] might be less if some input frames didn't yield a face
        # We need to match restored_faces with input_img_list, so length must be same as original video_length
        # This is tricky if face detection failed for some frames.
        # For now, assuming output_tensor.shape[0] matches number of successfully cropped_faces.
        # The pasting logic needs to handle this carefully.
        
        # Assuming restored_faces count matches cropped_faces count.
        restored_faces = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in output_tensor]
        del output_tensor 
        torch.cuda.empty_cache() 

    print('Pasting faces back ...')
    restored_frames_final_list = []
    # Iterate through original input_img_list to ensure output video has same number of frames
    # And use avg_landmarks that corresponds to original frame indices
    face_idx_counter = 0 # To map restored_faces to original frames where faces were found
    for i, img_original in enumerate(input_img_list):
        face_helper.clean_all()
        
        # Check if a face was successfully processed for this original frame 'i'
        # This requires knowing if avg_landmarks[i] was valid and resulted in a cropped_face
        # This is a simplification: assumes if avg_landmarks[i] is not NaN, a face was processed.
        # A more robust way would be to track indices of successfully cropped faces.
        is_face_processed_for_this_frame = not np.isnan(avg_landmarks[i]).any() if not args.has_aligned else True

        if not is_face_processed_for_this_frame or face_idx_counter >= len(restored_faces):
            # If no face was processed or we've run out of restored_faces (shouldn't happen if logic is tight)
            # Use original frame or a black frame. For now, use original.
            print(f"Warning: No restored face for frame {i}. Using original frame.")
            if args.bg_enhancement and bg_upsampler_instance: # Still apply BG enhancement if requested
                restored_frames_final_list.append(bg_upsampler_instance.enhance(img_original, outscale=args.upscale)[0])
            else:
                restored_frames_final_list.append(img_original)
            continue

        current_restored_face = restored_faces[face_idx_counter]
        face_idx_counter += 1

        if args.has_aligned: 
            img_resized_for_pasting = cv2.resize(img_original, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img_resized_for_pasting, threshold=10)
            if face_helper.is_gray: print('Grayscale input: True')
            face_helper.cropped_faces = [img_resized_for_pasting] # Provide the background
            face_helper.add_restored_face(current_restored_face.astype('uint8'))
            # For aligned, paste onto this resized image.
            pasted_img = face_helper.paste_faces_to_input_image(upsample_img=img_resized_for_pasting, draw_box=args.draw_box)
        else:
            face_helper.read_image(img_original) # Use original image for pasting
            face_helper.all_landmarks_5 = [avg_landmarks[i]] # Use landmarks for this specific frame
            # No align_warp_face here, that was for cropping. We need to paste back.
            face_helper.add_restored_face(current_restored_face.astype('uint8')) # Add the processed face
            
            bg_img_upsampled_for_paste = None
            if bg_upsampler_instance is not None:
                bg_img_upsampled_for_paste = bg_upsampler_instance.enhance(img_original, outscale=args.upscale)[0]
            
            face_helper.get_inverse_affine(None) 
            
            if args.face_upsample and face_upsampler_instance is not None:
                pasted_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img_upsampled_for_paste, draw_box=args.draw_box, face_upsampler=face_upsampler_instance)
            else:
                pasted_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img_upsampled_for_paste, draw_box=args.draw_box)
        
        if pasted_img is None:
            print(f"Warning: pasted_img is None for frame {i}. Using original frame as fallback.")
            restored_frames_final_list.append(img_original)
        else:
            restored_frames_final_list.append(pasted_img)

    print('Saving video ...')
    if not restored_frames_final_list:
        raise ValueError("No frames to save after processing.")
        
    height, width = restored_frames_final_list[0].shape[:2]
    output_video_filename = f"{clip_name}_enhanced.mp4"
    save_restore_path = os.path.join(output_dir, output_video_filename)
    
    vidwriter = VideoWriter(save_restore_path, height, width, fps)
    for f_idx, f_frame in enumerate(restored_frames_final_list):
        if f_frame is None: # Should be caught by above check, but defensive
            f_frame = np.zeros((height, width, 3), dtype=np.uint8)
        vidwriter.write_frame(f_frame)
    vidwriter.close()
    
    print(f'All results saved in {save_restore_path}.')
    return save_restore_path

def handler(job):
    print("Received job:", job)
    job_input = job.get('input', {})
    if not job_input:
        return {"error": "No input provided in the job."}

    video_url = job_input.get('video_url')
    if not video_url:
        return {"error": "Missing 'video_url' in job input."}

    print(f"Received video_url: {video_url}")
    job_id = job.get("id", "temp_video_job") 
    # Sanitize job_id for use in paths if necessary, or use a UUID
    # For simplicity, assume job_id is safe or RunPod provides a safe one.
    base_tmp_dir = "/tmp/runpod_jobs/" # Use /tmp for temporary files
    os.makedirs(base_tmp_dir, exist_ok=True)

    local_video_path = os.path.join(base_tmp_dir, f"{job_id}_downloaded.mp4")
    # Output dir also in /tmp, RunPod handles persistence if configured for output
    output_dir = os.path.join(base_tmp_dir, f"{job_id}_output/") 
    os.makedirs(output_dir, exist_ok=True)


    try:
        print(f"Downloading video from {video_url} to {local_video_path}...")
        response = requests.get(video_url, stream=True, timeout=60) # Added timeout
        response.raise_for_status()
        with open(local_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully to {local_video_path}")

        print("Starting video enhancement process...")
        processed_video_path = enhance_video_file(local_video_path, output_dir, job_input)
        print(f"Video enhancement complete. Output: {processed_video_path}")
        
        # The processed_video_path is what RunPod will serve or expect as output.
        # Ensure it's a path accessible within the container.
        return {
            "success": True,
            "message": "Video processed successfully.",
            "processed_video_path": processed_video_path 
        }

    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")
        return {"error": f"Failed to download video. Error: {str(e)}"}
    except FileNotFoundError as e:
        print(f"Error during processing (file not found): {e}")
        return {"error": f"File not found during processing. Error: {str(e)}"}
    except TypeError as e: 
        print(f"Type error during processing: {e}")
        return {"error": f"Type error during processing: {str(e)}"}
    except ValueError as e: # Catch ValueErrors from processing logic
        print(f"ValueError during processing: {e}")
        return {"error": f"ValueError during processing: {str(e)}"}
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc()) 
        return {"error": f"An unexpected error occurred. Error: {str(e)}"}
    finally:
        if os.path.exists(local_video_path):
            try:
                os.remove(local_video_path)
                print(f"Cleaned up downloaded file: {local_video_path}")
            except Exception as e_clean:
                print(f"Error cleaning up file {local_video_path}: {e_clean}")
        # Consider cleaning up output_dir if it's not automatically handled by RunPod for job outputs
        # For now, leave it, as RunPod might need it to serve the result.
        # if os.path.exists(output_dir):
        #    try:
        #        import shutil
        #        shutil.rmtree(output_dir)
        #        print(f"Cleaned up output directory: {output_dir}")
        #    except Exception as e_clean_dir:
        #        print(f"Error cleaning up directory {output_dir}: {e_clean_dir}")


if __name__ == '__main__':
    print("Starting RunPod worker for KEEP model via __main__...")
    runpod.serverless.start({"handler": handler})
