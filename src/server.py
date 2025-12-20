
import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
import uuid

import cv2
import numpy as np
import tensorflow as tf
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRelay
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import base64

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.style_transfer.model import make_style_transfer_network

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables
pcs = set()
relay = MediaRelay()
style_model = None

# Configuration
MODEL_PATH = "src/models/exported/sketch_20251219_loss699623.h5"  # Default hardcoded for MVP
INPUT_SHAPE = (None, None, 3)

def load_style_model():
    global style_model
    if style_model is None:
        logger.info(f"Loading Style Model from {MODEL_PATH}...")
        try:
            # Dynamic input shape for flexible resolution
            model = make_style_transfer_network(input_shape=INPUT_SHAPE)
            # Initialize dummy
            model(tf.zeros((1, 256, 256, 3)))
            model.load_weights(MODEL_PATH)
            style_model = model
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

class StyleTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames using a Style Transfer model.
    """
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.start_time = time.time()
        
        # Concrete function for faster inference? 
        # For variable resolution, we might rely on the dynamic graph or cache concrete functions per resolution.
        # For MVP, let's stick to eager-ish execution or simple @tf.function wrapped model.
        
    async def recv(self):
        frame = await self.track.recv()
        
        # Convert to numpy (YUV/RGB)
        # aiortc VideoFrame is typically YUV420P. to_ndarray(format="bgr24") converts it.
        img = frame.to_ndarray(format="bgr24")
        
        # Prepare for Model
        # CV2 BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, 0) # (1, H, W, 3)
        
        # Inference
        if style_model:
            # Run inference
            # Note: Running TF in async loop might block. 
            # Ideally should run in a separate thread/executor if it's slow.
            # But 3090 is fast enough for <10ms, so maybe okay directly.
            output_tensor = style_model(img_tensor, training=False)
            
            # Postprocess
            output_tensor = tf.clip_by_value(output_tensor, 0.0, 255.0)
            output_img = output_tensor[0].numpy().astype(np.uint8)
            
            # RGB -> BGR
            new_frame_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        else:
            new_frame_img = img

        # Rebuild VideoFrame
        from av import VideoFrame
        new_frame = VideoFrame.from_ndarray(new_frame_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        # Update Stats
        self.frame_count += 1
        return new_frame

@app.on_event("startup")
async def startup_event():
    load_style_model()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    logger.info("Created for %s", request.client.host)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received", track.kind)

        if track.kind == "video":
            # Hook up the style transfer track
            local_video = StyleTransformTrack(track)
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)
            # await pc.close()

    # Handle Offer
    await pc.setRemoteDescription(offer)
    
    # Create Answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            # Receive frame (expecting bytes or base64)
            data = await websocket.receive_text()
            
            # Assuming base64 encoded jpeg/png from canvas
            # Format: "data:image/jpeg;base64,....."
            if "," in data:
                header, encoded = data.split(",", 1)
            else:
                encoded = data
                
            # Decode
            image_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Inference
                # CV2 BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
                img_tensor = tf.expand_dims(img_tensor, 0)
                
                # Model
                if style_model:
                     output_tensor = style_model(img_tensor, training=False)
                     output_tensor = tf.clip_by_value(output_tensor, 0.0, 255.0)
                     output_img = output_tensor[0].numpy().astype(np.uint8)
                     img_out = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                else:
                     img_out = img
                
                # Encode back to JPEG
                _, buffer = cv2.imencode('.jpg', img_out)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # Send back
                await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>WebRTC Style Transfer</title>
</head>
<body>
    <h2>Style Transfer Test</h2>
    <div style="margin-bottom: 20px;">
        <button onclick="startWebRTC()">Start WebRTC (Best Quality)</button>
        <button onclick="startWebSocket()">Start WebSocket (Tunnel Safe)</button>
        <button onclick="stop()">Stop</button>
    </div>
    
    <div style="display: flex; gap: 10px;">
        <div style="position: relative;">
             <h3>Input (Camera)</h3>
             <video id="videoInput" autoplay playsinline muted style="width: 320px; border: 1px solid black; transform: scaleX(-1);"></video>
             <!-- Hidden Canvas for WebSocket Frame Capture -->
             <canvas id="canvasInput" width="320" height="240" style="display:none;"></canvas>
        </div>
        <div>
             <h3>Output (Server)</h3>
             <video id="videoOutput" autoplay playsinline style="width: 320px; border: 1px solid black; display:none;"></video>
             <img id="imageOutput" style="width: 320px; height: 240px; border: 1px solid black; display:none; background: #eee;" />
        </div>
    </div>
    
    <div id="status" style="margin-top: 10px; color: red;"></div>

    <script>
    var pc = null;
    var ws = null;
    var wsInterval = null;
    
    function stop() {
        if (pc) {
            pc.close();
            pc = null;
        }
        if (ws) {
            ws.close();
            ws = null;
        }
        if (wsInterval) {
            clearInterval(wsInterval);
            wsInterval = null;
        }
        document.getElementById('videoOutput').style.display = 'none';
        document.getElementById('imageOutput').style.display = 'none';
    }

    // --- WebSocket Mode (Safe Fallback) ---
    function startWebSocket() {
        stop();
        document.getElementById('status').innerText = "Connecting via WebSocket...";
        
        var videoIn = document.getElementById('videoInput');
        var canvasIn = document.getElementById('canvasInput');
        var ctx = canvasIn.getContext('2d');
        var imgOut = document.getElementById('imageOutput');
        
        imgOut.style.display = 'block';
        
        // 1. Start Camera
        navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } }).then(function(stream) {
            videoIn.srcObject = stream;
            
            // 2. Connect WS
            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            // Handle specific port forwarding scenario where host might be localhost:8000
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
            
            ws.onopen = function() {
                document.getElementById('status').innerText = "WebSocket Connected! Streaming...";
                
                // 3. Loop: Capture & Send
                wsInterval = setInterval(function() {
                    if (ws.readyState === WebSocket.OPEN) {
                        ctx.drawImage(videoIn, 0, 0, 320, 240);
                        var data = canvasIn.toDataURL('image/jpeg', 0.7); // Quality 0.7
                        ws.send(data);
                    }
                }, 100); // 10 FPS target for upload
            };
            
            ws.onmessage = function(evt) {
                // Receive frame
                imgOut.src = evt.data;
            };
            
            ws.onerror = function(e) { console.error(e); };
            
        }).catch(function(e) {
            document.getElementById('status').innerText = "Camera Error: " + e;
        });
    }

    // --- WebRTC Mode ---
    function startWebRTC() {
        stop();
        document.getElementById('status').innerText = "Connecting via WebRTC...";
        document.getElementById('videoOutput').style.display = 'block';
        
        var config = {
            sdpSemantics: 'unified-plan',
            iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
        };
        pc = new RTCPeerConnection(config);

        pc.addEventListener('track', function(evt) {
            if (evt.track.kind == 'video') {
                document.getElementById('videoOutput').srcObject = evt.streams[0];
                document.getElementById('status').innerText = "WebRTC Streaming!";
            }
        });

        navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then(function(stream) {
            document.getElementById('videoInput').srcObject = stream;
            stream.getTracks().forEach(track => pc.addTrack(track, stream));
            return pc.createOffer();
        }).then(function(offer) {
            return pc.setLocalDescription(offer);
        }).then(function() {
            return new Promise(function(resolve) {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    function checkState() {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icecandidate', checkState);
                            resolve();
                        }
                    }
                    pc.addEventListener('icecandidate', checkState);
                }
            });
        }).then(function() {
            var offer = pc.localDescription;
            return fetch('/offer', {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                }),
                headers: { 'Content-Type': 'application/json' },
                method: 'POST'
            });
        }).then(function(response) {
            return response.json();
        }).then(function(answer) {
            return pc.setRemoteDescription(answer);
        }).catch(function(e) {
            document.getElementById('status').innerText = e;
        });
    }
    </script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
