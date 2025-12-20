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
import asyncio

sys.path.insert(0, str(Path(__file__).parent / "style_transfer"))
from model import make_style_transfer_network

sys.path.insert(0, str(Path(__file__).parent))
from video_demo import process_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Style Transfer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
STYLES = ["sketch", "cyberpunk", "picasso", "vangogh"]

MODEL_DIR = Path(__file__).parent / "models" / "exported"
STYLE_DIR = Path(__file__).parent / "styles"


def load_models():
    """Load all style transfer models at startup"""
    global models, STYLES
    
    logger.info("Loading models...")
    
    if not MODEL_DIR.exists():
        logger.warning(f"Models directory not found: {MODEL_DIR}")
        logger.info("Using dummy models for demo purposes")
        return
    
    model_files = list(MODEL_DIR.glob("*.h5"))
    
    if not model_files:
        logger.warning("No model files found in models directory")
        return
    
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
            import traceback
            traceback.print_exc()
    
    if loaded_styles:
        STYLES = loaded_styles
        logger.info(f"Loaded {len(models)} models: {', '.join(STYLES)}")
    else:
        logger.warning("No models loaded. Using demo mode.")


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


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()
    logger.info("Server started successfully!")


@app.get("/")
async def root():
    return {
        "message": "AI Style Transfer API",
        "version": "1.0.0",
        "status": "running",
        "available_styles": STYLES,
        "models_loaded": list(models.keys())
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


@app.get("/test_video_imports")
async def test_video_imports():
    """Test if video processing imports work"""
    results = {
        "cv2_available": False,
        "process_video_available": False,
        "model_available": False
    }
    
    try:
        import cv2
        results["cv2_available"] = True
        results["cv2_version"] = cv2.__version__
    except Exception as e:
        results["cv2_error"] = str(e)
    
    try:
        from video_demo import process_video
        results["process_video_available"] = True
    except Exception as e:
        results["process_video_error"] = str(e)
    
    try:
        from style_transfer.model import make_style_transfer_network
        results["model_available"] = True
    except Exception as e:
        results["model_error"] = str(e)
    
    return results


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
        
        model_files = list(MODEL_DIR.glob(f"{style_id}_*.h5"))
        
        if model_files:
            model_path = str(model_files[0])
            logger.info(f"Using trained model: {model_path}")
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
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


if __name__ == "__main__":
    import uvicorn
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
