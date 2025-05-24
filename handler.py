import runpod
import os
import json
import requests
import cv2
import argparse
import glob
import torch
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device as basicsr_get_device # Renamed to avoid conflict
from scipy.ndimage import gaussian_filter1d
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.video_util import VideoReader, VideoWriter
from basicsr.utils.registry import ARCH_REGISTRY

# --- Global variables to be initialized by init() ---
device = None
keep_net = None
FACELIB_WEIGHTS_DIR = None
# Store functions in global scope if they depend on init's device or other init'd vars,
# or if they are just helpers defined within init and need to be accessed by handler.
# For simplicity, helper functions like interpolate_sequence can be defined globally if they don't depend on init state.
# set_realesrgan_upsampler depends on device and WEIGHTS_DIR, so it's better to make it accessible after init.
fn_set_realesrgan_upsampler = None 

# This helper can remain global if it's self-contained
def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)
    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))
        interpolated_sequence[missing_indices] = np.interp(x[missing_indices], x[valid_indices], sequence[valid_indices])
    return interpolated_sequence

MODEL_BASE_URL = "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/"
WEIGHTS_DIR = "/app/weights/" # Standard directory within the Docker image

def init():
    global device, keep_net, FACELIB_WEIGHTS_DIR, fn_set_realesrgan_upsampler

    print("[INIT] Starting model initialization...")
    
    print("[INIT] Getting device...")
    device = basicsr_get_device() 
    print(f"[INIT] Using device: {device}")

    print("[INIT] Creating directories...")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"[INIT] Ensured WEIGHTS_DIR exists: {WEIGHTS_DIR}")
    keep_dir = os.path.join(WEIGHTS_DIR, "KEEP")
    os.makedirs(keep_dir, exist_ok=True)
    print(f"[INIT] Ensured KEEP_DIR exists: {keep_dir}")
    
    FACELIB_WEIGHTS_DIR_local = os.path.join(WEIGHTS_DIR, "facelib")
    os.makedirs(FACELIB_WEIGHTS_DIR_local, exist_ok=True)
    print(f"[INIT] Ensured FACELIB_DIR exists: {FACELIB_WEIGHTS_DIR_local}")
    FACELIB_WEIGHTS_DIR = FACELIB_WEIGHTS_DIR_local # Assign to global

    realesrgan_dir = os.path.join(WEIGHTS_DIR, "realesrgan")
    os.makedirs(realesrgan_dir, exist_ok=True)
    print(f"[INIT] Ensured REALESRGAN_DIR exists: {realesrgan_dir}")
    print("[INIT] Directory creation complete.")

    # --- KEEP Model ---
    print("[INIT] Configuring KEEP model...")
    keep_model_config = {
        'architecture': {
            'img_size': 512, 'emb_dim': 256, 'dim_embd': 512, 'n_head': 8, 'n_layers': 9,
            'codebook_size': 1024, 'cft_list': ['16', '32', '64'], 'kalman_attn_head_dim': 48,
            'num_uncertainty_layers': 3, 'cfa_list': ['16', '32'], 'cfa_nhead': 4, 'cfa_dim': 256, 'cond': 1
        },
        'checkpoint_url': os.path.join(MODEL_BASE_URL, 'KEEP-b76feb75.pth'),
        'checkpoint_dir': keep_dir
    }
    print("[INIT] KEEP model configuration set.")

    print("[INIT] Initializing KEEP architecture...")
    loaded_keep_net = ARCH_REGISTRY.get('KEEP')(**keep_model_config['architecture']).to(device)
    print("[INIT] KEEP architecture initialized on device.")

    print(f"[INIT] Downloading KEEP model weights from {keep_model_config['checkpoint_url']} to {keep_model_config['checkpoint_dir']}...")
    keep_ckpt_path = load_file_from_url(url=keep_model_config['checkpoint_url'], model_dir=keep_model_config['checkpoint_dir'], progress=True, file_name=None)
    print(f"[INIT] KEEP model weights downloaded to {keep_ckpt_path}.")

    print("[INIT] Loading KEEP model checkpoint...")
    keep_checkpoint = torch.load(keep_ckpt_path, map_location=device, weights_only=True)
    print("[INIT] KEEP model checkpoint loaded.")

    print("[INIT] Loading state dict into KEEP model...")
    loaded_keep_net.load_state_dict(keep_checkpoint['params_ema'])
    print("[INIT] State dict loaded into KEEP model.")

    loaded_keep_net.eval()
    keep_net = loaded_keep_net # Assign to global
    print("[INIT] KEEP model loaded and set to eval mode.")

    # --- RealESRGAN Setup ---
    print("[INIT] Defining RealESRGAN upsampler factory function...")
    def _set_realesrgan_upsampler_internal():
        print("[INIT_RealESRGAN] Factory called. Initializing RealESRGAN...")
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer
        
        use_half = False
        if device.type == 'cuda':
            use_half = True 
        print(f"[INIT_RealESRGAN] use_half set to: {use_half}")

        realesrgan_model_url = os.path.join(MODEL_BASE_URL, 'RealESRGAN_x2plus.pth')
        print(f"[INIT_RealESRGAN] Downloading RealESRGAN model from {realesrgan_model_url} to {realesrgan_dir}...")
        model_path = load_file_from_url(
            url=realesrgan_model_url,
            model_dir=realesrgan_dir, progress=True, file_name=None
        )
        print(f"[INIT_RealESRGAN] RealESRGAN model downloaded to {model_path}.")

        print("[INIT_RealESRGAN] Initializing RRDBNet model...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        print("[INIT_RealESRGAN] RRDBNet model initialized.")

        print("[INIT_RealESRGAN] Initializing RealESRGANer...")
        upsampler = RealESRGANer(scale=2, model_path=model_path, model=model, tile=400, tile_pad=40, pre_pad=0, half=use_half, device=device)
        print("[INIT_RealESRGAN] RealESRGANer initialized.")

        if device.type == 'cpu':
            import warnings
            warnings.warn('[INIT_RealESRGAN] RealESRGAN: Running on CPU now!', category=RuntimeWarning)
        print("[INIT_RealESRGAN] RealESRGAN upsampler setup complete.")
        return upsampler
    
    fn_set_realesrgan_upsampler = _set_realesrgan_upsampler_internal
    print("[INIT] RealESRGAN upsampler factory function defined and assigned globally.")

    # --- Facelib Weights Download ---
    print("[INIT] Downloading Facelib helper models...")
    facelib_models_to_download = [
        'detection_Resnet50_Final.pth', 'detection_mobilenet0.25_Final.pth',
        'yolov5n-face.pth', 'yolov5l-face.pth', 'parsing_parsenet.pth'
    ]
    for model_name in facelib_models_to_download:
        url = os.path.join(MODEL_BASE_URL, model_name)
        print(f"[INIT_Facelib] Downloading {model_name} from {url} to {FACELIB_WEIGHTS_DIR}...")
        load_file_from_url(url=url, model_dir=FACELIB_WEIGHTS_DIR, progress=True, file_name=None)
        print(f"[INIT_Facelib] Downloaded {model_name}.")
    print("[INIT] Facelib helper models download complete.")
    
    print("[INIT] Model initialization (init function) complete.")
    # No return value needed as using globals

# --- enhance_video_file function (uses global model variables) ---
# This function's definition remains largely the same,
# but it will now implicitly use global `device`, `keep_net`, `FACELIB_WEIGHTS_DIR`, 
# and `fn_set_realesrgan_upsampler`.
# Ensure all references to these are direct (e.g., `keep_net` not `context.keep_net`)

def enhance_video_file(input_video_path, output_dir, job_input):
    # Uses global: device, keep_net (as net), FACELIB_WEIGHTS_DIR, fn_set_realesrgan_upsampler
    # Uses global helper: interpolate_sequence

    args = argparse.Namespace(
        input_path=input_video_path, 
        upscale=1, 
        max_processing_length=job_input.get('max_processing_length', 200),
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

    bg_upsampler_instance = None
    if args.bg_enhancement:
        bg_upsampler_instance = fn_set_realesrgan_upsampler() # Call the global function
    
    face_upsampler_instance = None
    if args.face_upsample:
        face_upsampler_instance = bg_upsampler_instance if bg_upsampler_instance is not None else fn_set_realesrgan_upsampler() # Call the global function

    net = keep_net # Use global keep_net
    
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
        device=device, # Use global device
        model_rootpath=FACELIB_WEIGHTS_DIR # Use global FACELIB_WEIGHTS_DIR
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
            # Pass detect_condition to ensure it's a number
            num_det_faces = face_helper.get_face_landmarks_5(only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5, only_keep_largest=True, detect_condition=0)

            if num_det_faces >= 1: # If any face is detected
                 raw_landmarks.append(face_helper.all_landmarks_5[0].reshape((10,)))
            else: # num_det_faces == 0
                 raw_landmarks.append(np.array([np.nan]*10))
        
        raw_landmarks = np.array(raw_landmarks)
        for i_lm in range(raw_landmarks.shape[1]): 
            raw_landmarks[:, i_lm] = interpolate_sequence(raw_landmarks[:, i_lm]) # Uses global interpolate_sequence
        video_length_raw = len(input_img_list) # Renamed to avoid conflict
        avg_landmarks = gaussian_filter1d(raw_landmarks, 5, axis=0).reshape(video_length_raw, 5, 2)
    
    cropped_faces = []
    for i, img in enumerate(input_img_list):
        face_helper.clean_all()
        face_helper.read_image(img)
        if not args.has_aligned: 
             face_helper.all_landmarks_5 = [avg_landmarks[i]]
        
        face_helper.align_warp_face() 
        if not face_helper.cropped_faces:
            print(f"Warning: No face cropped for frame {i}. Using a black placeholder.")
            # Ensure placeholder tensor is on the correct device
            cropped_faces.append(torch.zeros(3, 512, 512, dtype=torch.float32, device=device) * -1) 
            continue

        cropped_face_t = img2tensor(face_helper.cropped_faces[0] / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_faces.append(cropped_face_t) # Appending tensor, not list
        
    if not cropped_faces: 
        raise ValueError("Could not crop any faces from the video.")

    # Make sure all items in cropped_faces are tensors before stacking
    if not all(isinstance(cf, torch.Tensor) for cf in cropped_faces):
        raise TypeError("Not all cropped faces are tensors, check placeholder logic.")

    cropped_faces_tensor = torch.stack(cropped_faces, dim=0).unsqueeze(0).to(device) # Renamed variable
    
    print('Restoring faces ...')
    with torch.no_grad():
        video_length_restored = cropped_faces_tensor.shape[1] # Renamed
        output_frames_list = [] 
        for start_idx in range(0, video_length_restored, args.max_processing_length): # Use max_processing_length
            end_idx = min(start_idx + args.max_processing_length, video_length_restored) # Use max_processing_length
            current_segment_input = cropped_faces_tensor[:, start_idx:end_idx, ...]
            
            if current_segment_input.shape[1] == 0: 
                continue
            
            # Adjusted logic for single frame segments
            if current_segment_input.shape[1] == 1:
                # If it's a single frame segment, duplicate it to form a sequence of 2, then take the first output.
                # This is a common trick if models expect sequences.
                # However, the KEEP model might handle single frames if its architecture supports it.
                # The original logic was:
                # if end_idx - start_idx == 1: output.append(net(cropped_faces[:, [start_idx, start_idx], ...])[:, 0:1, ...])
                # This implies duplicating the frame if the segment length is 1.
                duplicated_input = current_segment_input.repeat(1,2,1,1,1)
                output_segment = net(duplicated_input, need_upscale=False)[:, 0:1, ...] # Take first frame output
            else:
                output_segment = net(current_segment_input, need_upscale=False)
            output_frames_list.append(output_segment)
        
        if not output_frames_list:
             raise ValueError("No output frames generated by the KEEP model.")

        output_tensor = torch.cat(output_frames_list, dim=1).squeeze(0) 
        
        restored_faces = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in output_tensor]
        del output_tensor 
        torch.cuda.empty_cache() 

    print('Pasting faces back ...')
    restored_frames_final_list = []
    face_idx_counter = 0 
    for i, img_original in enumerate(input_img_list):
        face_helper.clean_all()
        
        is_face_processed_for_this_frame = False
        if not args.has_aligned:
            # Check if avg_landmarks[i] was valid (not NaN) AND resulted in a successfully cropped face earlier
            # This requires that cropped_faces list corresponds index-wise to input_img_list for placeholders
            if i < len(avg_landmarks) and not np.isnan(avg_landmarks[i]).any():
                # Additionally, check if the corresponding entry in cropped_faces was not a placeholder
                # This check is imperfect as the placeholder is also a tensor.
                # A better way would be to have a parallel list of booleans indicating success of cropping.
                # For now, assume if landmarks were not NaN, a face was attempted.
                is_face_processed_for_this_frame = True 
        else: # has_aligned == True
            is_face_processed_for_this_frame = True


        if not is_face_processed_for_this_frame or face_idx_counter >= len(restored_faces):
            print(f"Warning: No restored face for frame {i} or ran out of restored faces. Using original frame.")
            if args.bg_enhancement and bg_upsampler_instance: 
                restored_frames_final_list.append(bg_upsampler_instance.enhance(img_original, outscale=args.upscale)[0])
            else:
                restored_frames_final_list.append(img_original)
            if not is_face_processed_for_this_frame: # Only skip incrementing face_idx_counter if this frame was skipped for face processing
                pass # Do not increment face_idx_counter
            else: # Ran out of restored faces (should not happen)
                face_idx_counter +=1

            continue

        current_restored_face = restored_faces[face_idx_counter]
        face_idx_counter += 1

        if args.has_aligned: 
            img_resized_for_pasting = cv2.resize(img_original, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img_resized_for_pasting, threshold=10)
            if face_helper.is_gray: print('Grayscale input: True')
            face_helper.cropped_faces = [img_resized_for_pasting] 
            face_helper.add_restored_face(current_restored_face.astype('uint8'))
            pasted_img = face_helper.paste_faces_to_input_image(upsample_img=img_resized_for_pasting, draw_box=args.draw_box)
        else:
            face_helper.read_image(img_original) 
            face_helper.all_landmarks_5 = [avg_landmarks[i]] 
            face_helper.add_restored_face(current_restored_face.astype('uint8')) 
            
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
        if f_frame is None: 
            print(f"Warning: Frame {f_idx} is None during saving. Using black frame.")
            f_frame = np.zeros((height, width, 3), dtype=np.uint8)
        vidwriter.write_frame(f_frame)
    vidwriter.close()
    
    print(f'All results saved in {save_restore_path}.')
    return save_restore_path


# --- Main Handler ---
def handler(job):
    # This function now assumes init() has been called and populated globals:
    # device, keep_net, FACELIB_WEIGHTS_DIR, fn_set_realesrgan_upsampler
    # It also uses global helper: interpolate_sequence (implicitly via enhance_video_file)

    print("Received job:", job)
    job_input = job.get('input', {})
    if not job_input:
        return {"error": "No input provided in the job."}

    video_url = job_input.get('video_url')
    if not video_url:
        return {"error": "Missing 'video_url' in job input."}

    print(f"Received video_url: {video_url}")
    job_id = job.get("id", "temp_video_job") 
    
    base_tmp_dir = "/tmp/runpod_jobs/" 
    os.makedirs(base_tmp_dir, exist_ok=True)
    local_video_path = os.path.join(base_tmp_dir, f"{job_id}_downloaded.mp4")
    output_dir = os.path.join(base_tmp_dir, f"{job_id}_output/") 
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Downloading video from {video_url} to {local_video_path}...")
        response = requests.get(video_url, stream=True, timeout=60) 
        response.raise_for_status()
        with open(local_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully to {local_video_path}")

        print("Starting video enhancement process...")
        # Ensure enhance_video_file is called correctly
        processed_video_path = enhance_video_file(local_video_path, output_dir, job_input)
        print(f"Video enhancement complete. Output: {processed_video_path}")
        
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
    except ValueError as e: 
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
        # Note: output_dir is not cleaned up, assuming RunPod handles it or it's desired.


if __name__ == '__main__':
    print("Starting RunPod worker for KEEP model with init function...")
    runpod.serverless.start({
        "handler": handler,
        "init": init  # Pass the init function to RunPod
    })
