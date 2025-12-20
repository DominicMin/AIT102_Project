from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
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

sys.path.insert(0, str(Path(__file__).parent / "style_transfer"))
from model import make_style_transfer_network

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
    
    # Find all .h5 model files
    model_files = list(MODEL_DIR.glob("*.h5"))
    
    if not model_files:
        logger.warning("No model files found in models directory")
        return
    
    loaded_styles = []
    
    for model_path in model_files:
        # Extract style name from filename (e.g., sketch_20251219_loss699623.h5 -> sketch)
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
        "available_styles": STYLES
    }


if __name__ == "__main__":
    import uvicorn
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
