from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from pathlib import Path
import logging
import sys
import cv2
import tempfile
import traceback
import threading
import time
import onnxruntime as ort
import asyncio

sys.path.insert(0, str(Path(__file__).parent / "style_transfer"))
from model import make_style_transfer_network

sys.path.insert(0, str(Path(__file__).parent))
from video_demo import process_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    load_models() # Load TF models
    load_god_mode_models() # Load ONNX models
    logger.info("Server started successfully!")
    yield
    # Shutdown: Clean up resources if needed
    if stream_state.cap:
        stream_state.cap.release()
    logger.info("Server shutting down...")

# --- App Initialization (MUST BE EARLY) ---
app = FastAPI(title="AI Style Transfer API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TensorFlow Global State ---
models = {}
STYLES = ["sketch", "cyberpunk", "picasso", "vangogh"]

MODEL_DIR = Path(__file__).parent / "models" / "exported"
STYLE_DIR = Path(__file__).parent / "styles"

# --- God Mode (ONNX) Global State ---
class StreamState:
    def __init__(self):
        self.sessions = {} # { 'sketch': session, 'vangogh': session ... }
        self.active_style = None
        self.active_session = None
        self.camera_id = 0
        self.width = 640
        self.fps = 0.0
        self.cap = None
        self.lock = threading.Lock()

stream_state = StreamState()

# --- TensorFlow Helper Functions ---
def load_models():
    """Load all style transfer models at startup"""
    global models, STYLES
    
    logger.info("Loading TensorFlow models...")
    
    if not MODEL_DIR.exists():
        logger.warning(f"Models directory not found: {MODEL_DIR}")
        logger.info("Using dummy models for demo purposes")
        return
    
    model_files = list(MODEL_DIR.glob("*.h5"))
    
    if not model_files:
        logger.warning("No .h5 model files found in models directory")
        # Don't return, we might find ONNX models
    
    loaded_styles = []
    
    for model_path in model_files:
        style_name = model_path.stem.split('_')[0]
        
        try:
            logger.info(f"Loading {style_name} model from {model_path.name}...")
            model = make_style_transfer_network(input_shape=(256, 256, 3))
            model.load_weights(str(model_path))
            
            models[style_name] = model
            loaded_styles.append(style_name)
            logger.info(f"✓ Successfully loaded {style_name} model!")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {style_name} model: {e}")
            # traceback.print_exc()
    
    if loaded_styles:
        # Merge existing styles with loaded ones to avoid duplicates
        STYLES = list(set(STYLES + loaded_styles))
        logger.info(f"Loaded {len(models)} TF models")
    else:
        logger.warning("No TF models loaded.")

def preprocess_image(image_bytes: bytes, target_size=(256, 256)):
    """Preprocess image for model input"""
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, image


def postprocess_image(output_array):
    """Convert model output to PIL Image"""
    output = np.squeeze(output_array, axis=0)
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return Image.fromarray(output)


def apply_style_transform(image_bytes: bytes, style_id: str):
    """Apply style transfer to an image"""
    img_array, original_size = preprocess_image(image_bytes)
    
    if style_id in models:
        try:
            output = models[style_id].predict(img_array, verbose=0)
            result_image = postprocess_image(output)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            result_image = Image.fromarray((img_array[0] * 255).astype(np.uint8))
    else:
        img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
        result_image = apply_demo_style(img, style_id)
    
    return result_image


def apply_demo_style(image: Image.Image, style_id: str):
    """Apply demo style transforms when models are not available"""
    import numpy as np
    
    img_array = np.array(image)
    
    if style_id == "picasso":
        img_array = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
    elif style_id == "vangogh":
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.3, 0, 255)
    elif style_id == "ukiyoe":
        sepia = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        img_array = np.dot(img_array, sepia.T)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    elif style_id == "cyberpunk":
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.3, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.3, 0, 255)
        img_array = img_array.astype(np.uint8)
    elif style_id == "sketch":
        gray = np.dot(img_array, [0.299, 0.587, 0.114])
        img_array = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    
    return Image.fromarray(img_array)

def process_video_with_demo_style(style_id: str, input_path: str, output_path: str, target_width: int = None):
    """Process video with demo style effects when trained model is not available"""
    logger.info(f"Starting demo style processing for: {input_path}")
    
    if not os.path.exists(input_path):
        raise Exception(f"Input video file not found: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video with cv2: {input_path}")
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        fps = 30.0
        logger.warning("FPS is 0, using default 30 FPS")
    
    logger.info(f"Video properties: {orig_width}x{orig_height}, {fps} FPS, {total_frames} frames")
    
    if target_width and target_width != orig_width:
        scale = target_width / orig_width
        new_width = target_width
        new_height = int(orig_height * scale)
    else:
        new_width = orig_width
        new_height = orig_height
    
    logger.info(f"Output dimensions: {new_width}x{new_height}")
    
    # Use temp file for initial processing
    temp_output = output_path.replace('.mp4', '_temp.mp4') if '.mp4' in output_path else output_path + '_temp'
    
    # Try H.264 codecs first for browser compatibility
    codecs_to_try = [
        ('avc1', 'H.264'),
        ('H264', 'H.264 alternative'),
        ('X264', 'x264'),
        ('mp4v', 'MPEG-4 fallback')
    ]
    
    out = None
    codec_used = None
    for codec, name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_out = cv2.VideoWriter(temp_output, fourcc, fps, (new_width, new_height))
            if test_out.isOpened():
                out = test_out
                codec_used = name
                logger.info(f"Using codec: {name} ({codec})")
                break
            else:
                test_out.release()
        except Exception as e:
            logger.warning(f"Codec {codec} not available: {e}")
    
    if out is None:
        raise Exception("No suitable video codec available")
    
    if not out.isOpened():
        raise Exception(f"Failed to create output video writer: {output_path}")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if target_width and target_width != orig_width:
                frame = cv2.resize(frame, (new_width, new_height))
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(img_rgb)
            output_pil = apply_demo_style(pil_img, style_id)
            output_img = np.array(output_pil)
            
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            out.write(output_bgr)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Processed frame {frame_count}/{total_frames} ({progress:.1f}%)")
    
    finally:
        cap.release()
        out.release()
    
    logger.info(f"Video processing complete: {frame_count} frames processed")


# --- ONNX Helper Functions ---
def init_onnx_session(model_path):
    logger.info(f"Loading ONNX Model: {model_path}")
    available_providers = ort.get_available_providers()
    providers = []
    if 'DmlExecutionProvider' in available_providers:
        providers.append('DmlExecutionProvider')
    elif 'DirectMLExecutionProvider' in available_providers:
        providers.append('DirectMLExecutionProvider')
    
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
        return session
    except Exception as e:
        logger.error(f"Error loading ONNX: {e}")
        return None

def load_god_mode_models():
    """Load ONNX models for Real-time God Mode"""
    logger.info(f"Scanning for ONNX models in {MODEL_DIR}...")
    
    # Use same directory as TF models
    model_dir = MODEL_DIR
    
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return

    known_styles = ["sketch", "vangogh", "picasso", "cyberpunk", "monet", "seurat", "wuguanzhong"]
    files = list(model_dir.glob("*"))
    logger.info(f"Found {len(files)} files in directory")

    loaded_count = 0
    
    # Standard Loading
    for style in known_styles:
        # Find best file for this style
        # glob return Path objects
        candidates = [f for f in files if f.name.startswith(style) and f.suffix == ".onnx"]
        
        if not candidates:
            continue
            
        # Prefer FP16
        fp16_candidates = [f for f in candidates if "_fp16.onnx" in f.name]
        best_file = fp16_candidates[0] if fp16_candidates else candidates[0]
        
        logger.info(f"Trying to load style '{style}' from {best_file.name}")
        session = init_onnx_session(best_file)
        
        if session:
            stream_state.sessions[style] = session
            logger.info(f"✓ Loaded ONNX style '{style}' from {best_file.name}")
            loaded_count += 1
            
            # Set first loaded as active if none set
            if stream_state.active_session is None:
                stream_state.active_session = session
                stream_state.active_style = style
                logger.info(f"Set initial active style to: {style}")

    # Fallback Loading
    if loaded_count == 0:
        logger.info("No standard ONNX styles found. Loading all .onnx files as 'custom'...")
        for f in files:
            if f.suffix == ".onnx":
                session = init_onnx_session(f)
                if session:
                    style_id = f.stem
                    stream_state.sessions[style_id] = session
                    logger.info(f"✓ Loaded custom ONNX style '{style_id}' from {f.name}")
                    loaded_count += 1
                    
        if loaded_count > 0 and stream_state.active_session is None:
            first_style = list(stream_state.sessions.keys())[0]
            stream_state.active_session = stream_state.sessions[first_style]
            stream_state.active_style = first_style
            logger.info(f"Set fallback active style to: {first_style}")

    logger.info(f"Total loaded ONNX models: {loaded_count}")
    logger.info(f"Available ONNX styles: {list(stream_state.sessions.keys())}")

# --- MJPEG Generator ---
def get_camera_frame():
    """Generator that yields MJPEG frames"""
    logger.info("Opening Global VideoCapture...")
    if stream_state.cap is None or not stream_state.cap.isOpened():
        stream_state.cap = cv2.VideoCapture(stream_state.camera_id)
        if not stream_state.cap.isOpened():
             logger.error(f"ERROR: Could not open camera {stream_state.camera_id}")
    
    prev_time = time.time()
    frame_count_log = 0
    
    while True:
        ret, frame = stream_state.cap.read()
        if not ret:
            logger.warning("Camera read failed. Retrying in 1s...")
            if stream_state.cap:
                stream_state.cap.release()
            time.sleep(1)
            stream_state.cap = cv2.VideoCapture(stream_state.camera_id)
            continue
            
        if stream_state.active_session is None:
            if frame_count_log % 30 == 0:
                logger.warning("No active session! Waiting...")
            time.sleep(0.1)
            frame_count_log += 1
            continue

        # Resize
        h, w = frame.shape[:2]
        if stream_state.width and w != stream_state.width:
            scale = stream_state.width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (stream_state.width, new_h))
            
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        try:
            input_node = stream_state.active_session.get_inputs()[0]
            input_name = input_node.name
            output_name = stream_state.active_session.get_outputs()[0].name
            
            # Check expected type (tensor(float) or tensor(float16))
            if "float16" in input_node.type:
                input_tensor = np.expand_dims(img_rgb.astype(np.float16), axis=0)
            else:
                input_tensor = np.expand_dims(img_rgb.astype(np.float32), axis=0)

            output_tensor = stream_state.active_session.run([output_name], {input_name: input_tensor})[0]
            
            # Postprocess
            output_tensor = np.clip(output_tensor, 0.0, 255.0).astype(np.uint8)
            output_img = output_tensor[0]
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            output_bgr = frame 

        # Stats
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        stream_state.fps = fps
        
        frame_count_log += 1
        if frame_count_log % 60 == 0:
             logger.info(f"Stream FPS: {fps:.2f}, Style: {stream_state.active_style}")
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', output_bgr)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Start Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "Unified AI Style Transfer API",
        "version": "2.0.0",
        "status": "running",
        "available_styles": STYLES,
        "onnx_styles": list(stream_state.sessions.keys()),
        "active_god_mode_style": stream_state.active_style
    }

@app.get("/styles")
async def get_styles():
    """Get list of available styles"""
    return {
        "styles": [
            {
                "id": style,
                "name": style.capitalize(),
                "preview_url": f"/static/previews/{style}.jpg"
            }
            for style in STYLES
        ]
    }

@app.post("/transform")
async def transform_single(file: UploadFile = File(...), style_id: str = "sketch"):
    """Transform a single image with sketch style"""
    if style_id not in STYLES:
        raise HTTPException(status_code=400, detail=f"Invalid style. Choose from: {STYLES}")
    
    try:
        image_bytes = await file.read()
        
        result_image = apply_style_transform(image_bytes, style_id)
        
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG', quality=95)
        output_buffer.seek(0)
        
        return Response(content=output_buffer.getvalue(), media_type="image/jpeg")
    
    except Exception as e:
        logger.error(f"Transform failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transform_all")
async def transform_all(file: UploadFile = File(...)):
    """Transform image with all available styles (parallel processing)"""
    try:
        image_bytes = await file.read()
        
        results = []

        for style_id in STYLES:
            try:
                result_image = apply_style_transform(image_bytes, style_id)
                
                output_buffer = io.BytesIO()
                result_image.save(output_buffer, format='JPEG', quality=90)
                img_base64 = base64.b64encode(output_buffer.getvalue()).decode()
                
                results.append({
                    "style_id": style_id,
                    "image_base64": img_base64
                })
                
                logger.info(f"✓ Processed style: {style_id}")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {style_id}: {e}")
        
        return JSONResponse(content={"results": results})
    
    except Exception as e:
        logger.error(f"Transform all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_styles": STYLES,
        "model_dir": str(MODEL_DIR),
        "model_dir_exists": MODEL_DIR.exists()
    }

@app.post("/transform_video")
async def transform_video(file: UploadFile = File(...), style_id: str = "sketch", width: int = None):
    """Transform a video with the selected style"""
    if style_id not in STYLES:
        raise HTTPException(status_code=400, detail=f"Invalid style. Choose from: {STYLES}")
    
    temp_input_path = None
    temp_output_path = None
    
    try:
        logger.info(f"Starting video transformation with style: {style_id}")
        logger.info(f"Input file: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input_path = temp_input.name
        temp_output_path = temp_output.name
        
        logger.info(f"Created temp files: input={temp_input_path}, output={temp_output_path}")
        
        video_bytes = await file.read()
        logger.info(f"Read {len(video_bytes)} bytes from uploaded file")
        
        temp_input.write(video_bytes)
        temp_input.close()
        
        logger.info(f"Saved video to temp file: {temp_input_path}")
        
        if style_id in models:
            logger.info(f"Using pre-loaded model for {style_id}")
            try:
                process_video(
                    model_path=None,
                    input_video=temp_input_path,
                    output_video=temp_output_path,
                    width=width,
                    loaded_model=models[style_id]
                )
                logger.info("Model processing completed successfully")
            except Exception as e:
                logger.error(f"Model processing failed: {e}, falling back to demo style")
                traceback.print_exc()
                process_video_with_demo_style(
                    style_id=style_id,
                    input_path=temp_input_path,
                    output_path=temp_output_path,
                    target_width=width
                )
        else:
            model_files = list(MODEL_DIR.glob(f"{style_id}_*.h5"))
            
            if model_files:
                model_path = str(model_files[0])
                logger.info(f"Using trained model from file: {model_path}")
                try:
                    process_video(
                        model_path=model_path,
                        input_video=temp_input_path,
                        output_video=temp_output_path,
                        width=width
                    )
                    logger.info("Model processing completed successfully")
                except Exception as e:
                    logger.error(f"Model processing failed: {e}, falling back to demo style")
                    traceback.print_exc()
                    process_video_with_demo_style(
                        style_id=style_id,
                        input_path=temp_input_path,
                        output_path=temp_output_path,
                        target_width=width
                    )
            else:
                logger.warning(f"No trained model found for {style_id}, using demo processing")
                process_video_with_demo_style(
                    style_id=style_id,
                    input_path=temp_input_path,
                    output_path=temp_output_path,
                    target_width=width
                )
        
        logger.info(f"Video processing completed, output at: {temp_output_path}")
        
        if not os.path.exists(temp_output_path):
            raise Exception(f"Output video file was not created: {temp_output_path}")
        
        output_size = os.path.getsize(temp_output_path)
        logger.info(f"Output video size: {output_size} bytes")
        
        if os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
                logger.info(f"Cleaned up input temp file")
            except Exception as e:
                logger.warning(f"Failed to cleanup input file: {e}")
        
        def cleanup_output():
            """Cleanup function to delete temp output file after sending"""
            try:
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                    logger.info(f"Cleaned up output temp file")
            except Exception as e:
                logger.error(f"Failed to cleanup output file: {e}")
        
        return FileResponse(
            temp_output_path,
            media_type="video/mp4",
            filename=f"styled_{style_id}_{file.filename}",
            background=BackgroundTasks().add_task(cleanup_output)
        )
    
    except Exception as e:
        error_msg = f"Video transform failed: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
                logger.info(f"Cleaned up input temp file after error")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup input file: {cleanup_error}")
        
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
                logger.info(f"Cleaned up output temp file after error")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup output file: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=error_msg)

# --- God Mode Endpoints ---

@app.get("/stats")
def get_stats():
    """Return current inference FPS and active style"""
    return {
        "fps": stream_state.fps,
        "style": stream_state.active_style
    }

@app.get("/switch_style")
def switch_style(style_id: str):
    if style_id in stream_state.sessions:
        stream_state.active_session = stream_state.sessions[style_id]
        stream_state.active_style = style_id
        logger.info(f"Switched to style: {style_id}")
        return {"status": "success", "style": style_id}
    else:
        return Response(content="Style not loaded", status_code=404)

@app.get("/video_feed")
def video_feed():
    if not stream_state.active_session:
        return Response(content="No models loaded", status_code=500)
    
    return StreamingResponse(get_camera_frame(), 
                             media_type='multipart/x-mixed-replace; boundary=frame')





if __name__ == "__main__":
    import uvicorn
    import argparse
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple argument parsing to allow overriding port or camera
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--port", type=int, default=8000, help="Server Port")
    
    # Use parse_known_args to avoid conflict if uvicorn passes other args
    args, unknown = parser.parse_known_args()
    
    stream_state.camera_id = args.camera
    
    logger.info(f"Starting Unified Server on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
