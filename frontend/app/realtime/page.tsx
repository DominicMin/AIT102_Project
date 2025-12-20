'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'

export default function RealtimePage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const [isStreaming, setIsStreaming] = useState(false)
  const [isGodMode, setIsGodMode] = useState(true) // Default to God Mode
  const [selectedStyle, setSelectedStyle] = useState('sketch')
  const [error, setError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)
  const [modelLoaded, setModelLoaded] = useState(false)

  const styles = [
    { id: 'sketch', name: 'Sketch', icon: '✏️' },
    { id: 'cyberpunk', name: 'Cyberpunk', icon: '🌃' },
    { id: 'picasso', name: 'Picasso', icon: '🎨' },
    { id: 'vangogh', name: 'Van Gogh', icon: '🌻' },
    { id: 'monet', name: 'Monet', icon: '🎨' },
    { id: 'seurat', name: 'Seurat', icon: '🖼️' },
    { id: 'wuguanzhong', name: 'Chinese Ink', icon: '🖌️' },
  ]

  useEffect(() => {
    return () => {
      stopStreaming()
    }
  }, [])

  const startStreaming = async () => {
    setError(null)

    if (isGodMode) {
      // --- GOD MODE (Backend Streaming) ---
      setIsStreaming(true)
      setModelLoaded(true)

      // Poll FPS from backend
      if (intervalRef.current) clearInterval(intervalRef.current)
      intervalRef.current = setInterval(async () => {
        try {
          const res = await fetch('http://localhost:8000/stats')
          if (res.ok) {
            const data = await res.json()
            setFps(Math.round(data.fps))
          }
        } catch (e) {
          console.error("FPS Poll Error", e)
        }
      }, 1000)

    } else {
      // --- WEB DEMO (Client-side Canvas) ---
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          setIsStreaming(true)
          setModelLoaded(true)
          processFrames()
        }
      } catch (err: any) {
        setError(`Unable to access camera: ${err.message}`)
        console.error('Camera access error:', err)
      }
    }
  }

  const stopStreaming = () => {
    setIsStreaming(false)
    setFps(0)

    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
  }

  const processFrames = () => {
    if (!isStreaming && videoRef.current?.srcObject) return

    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let lastFrameTime = performance.now()
    let frameCount = 0
    let fpsUpdateTime = performance.now()

    const drawFrame = () => {
      if (!isStreaming) return

      // If switched to God Mode mid-stream, stop canvas loop
      if (isGodMode) return

      if (video.videoWidth === 0) {
        requestAnimationFrame(drawFrame)
        return
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      applyDemoFilter(imageData, selectedStyle)
      ctx.putImageData(imageData, 0, 0)

      frameCount++
      const currentTime = performance.now()
      if (currentTime - fpsUpdateTime > 1000) {
        setFps(Math.round((frameCount * 1000) / (currentTime - fpsUpdateTime)))
        frameCount = 0
        fpsUpdateTime = currentTime
      }

      requestAnimationFrame(drawFrame)
    }

    drawFrame()
  }

  const applyDemoFilter = (imageData: ImageData, style: string) => {
    const data = imageData.data
    switch (style) {
      case 'sketch':
        for (let i = 0; i < data.length; i += 4) {
          const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114
          data[i] = data[i + 1] = data[i + 2] = 255 - gray
        }
        break
      case 'cyberpunk':
        for (let i = 0; i < data.length; i += 4) {
          data[i] = Math.min(255, data[i] * 0.8 + 50)
          data[i + 1] = Math.min(255, data[i + 1] * 1.2)
          data[i + 2] = Math.min(255, data[i + 2] * 1.4)
        }
        break
      case 'picasso':
        for (let i = 0; i < data.length; i += 4) {
          data[i] = data[i] > 128 ? 255 : 0
          data[i + 1] = data[i + 1] > 128 ? 255 : 0
          data[i + 2] = data[i + 2] > 128 ? 255 : 0
        }
        break
      case 'vangogh':
        for (let i = 0; i < data.length; i += 4) {
          data[i] = Math.min(255, data[i] * 1.3)
          data[i + 1] = Math.min(255, data[i + 1] * 1.1)
          data[i + 2] = Math.max(0, data[i + 2] * 0.7)
        }
        break
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 backdrop-blur-sm bg-black/30">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="text-gray-400 hover:text-white transition-colors">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </Link>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-600 bg-clip-text text-transparent">
                Real-time Style Transfer
              </h1>
            </div>
            <div className="flex items-center gap-4 text-base text-gray-400">
              <span className={`font-bold transition-colors ${isGodMode ? 'text-purple-400' : 'text-gray-500'}`}>
                {isGodMode ? '⚡ GOD MODE (ONNX)' : '🌐 Web Demo'}
              </span>
              {/* Toggle Switch */}
              <button
                onClick={() => {
                  stopStreaming()
                  setIsGodMode(!isGodMode)
                }}
                className={`w-14 h-7 rounded-full p-1 transition-colors ${isGodMode ? 'bg-purple-600' : 'bg-gray-600'}`}
              >
                <div className={`bg-white w-5 h-5 rounded-full shadow-md transform transition-transform ${isGodMode ? 'translate-x-7' : 'translate-x-0'}`} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-12">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-5xl mx-auto">

          {/* Info Banner */}
          <div className={`mb-8 p-6 rounded-xl border ${isGodMode ? 'bg-purple-900/20 border-purple-500/30' : 'bg-blue-900/20 border-blue-500/30'}`}>
            <div className="flex items-start gap-4">
              <div className="text-3xl">{isGodMode ? '⚡' : '🎥'}</div>
              <div>
                <h2 className="text-xl font-bold mb-2">
                  {isGodMode ? 'GOD MODE Active (RTX Accelerated)' : 'Web Demo Mode (CPU)'}
                </h2>
                <p className="text-gray-300 mb-2">
                  {isGodMode
                    ? 'Using backend ONNX runtime with DirectML acceleration. Models running on NVIDIA RTX 4060.'
                    : 'Using client-side Canvas API for basic filter demonstration.'}
                </p>
              </div>
            </div>
          </div>

          {/* Style Selection */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-4">Select Style</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {styles.map((style) => (
                <button
                  key={style.id}
                  onClick={() => {
                    setSelectedStyle(style.id)
                    if (isGodMode && isStreaming) {
                      fetch(`http://localhost:8000/switch_style?style_id=${style.id}`)
                        .catch(e => console.error("Failed to switch style", e))
                    }
                  }}
                  disabled={!isStreaming && !isGodMode}
                  className={`p-4 rounded-xl font-semibold text-lg transition-all ${selectedStyle === style.id
                    ? `bg-gradient-to-r ${isGodMode ? 'from-purple-600 to-pink-600' : 'from-blue-600 to-cyan-600'} shadow-lg scale-105`
                    : 'bg-gray-800 hover:bg-gray-700'
                    } ${(!isStreaming && !isGodMode) ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="text-2xl mb-2">{style.icon}</div>
                  <div>{style.name}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Video Display */}
          <div className="relative bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 shadow-2xl aspect-video">
            {/* Stats Overlay */}
            {isStreaming && (
              <div className="absolute top-4 right-4 z-10 bg-black/70 backdrop-blur-sm px-4 py-2 rounded-lg">
                <div className="text-sm">
                  <div className="text-green-400 font-bold">FPS: {fps}</div>
                  <div className="text-gray-400">Style: {selectedStyle}</div>
                </div>
              </div>
            )}

            {isStreaming ? (
              isGodMode ? (
                // GOD MODE STREAM
                <img
                  src="http://localhost:8000/video_feed"
                  className="w-full h-full object-contain"
                  alt="God Mode Live Stream"
                  onError={() => setError("Backend Connection Failed. Check server.py console (Port 8000).")}
                />
              ) : (
                // WEB DEMO STREAM
                <>
                  <video ref={videoRef} autoPlay playsInline muted className="hidden" />
                  <canvas ref={canvasRef} className="w-full h-full object-contain" />
                </>
              )
            ) : (
              // PLACEHOLDER
              <div className="w-full h-full flex items-center justify-center bg-gray-800">
                <div className="text-center">
                  <div className="text-6xl mb-4">🚀</div>
                  <p className="text-xl text-gray-400">Ready to Start</p>
                  <p className="text-sm text-gray-500 mt-2">
                    {isGodMode ? 'Ensure "python src/server.py" is running' : 'Browser permission required'}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200">
              ⚠️ {error}
            </div>
          )}

          {/* Controls */}
          <div className="mt-6 flex gap-4">
            {!isStreaming ? (
              <button
                onClick={startStreaming}
                className={`flex-1 px-8 py-4 bg-gradient-to-r ${isGodMode ? 'from-purple-600 to-pink-600' : 'from-blue-600 to-cyan-600'} rounded-full font-semibold text-lg hover:brightness-110 transition-all shadow-lg`}
              >
                {isGodMode ? '⚡ Start God Mode' : '🎬 Start Web Demo'}
              </button>
            ) : (
              <button
                onClick={stopStreaming}
                className="flex-1 px-8 py-4 bg-gradient-to-r from-red-600 to-orange-600 rounded-full font-semibold text-lg hover:from-red-700 hover:to-orange-700 transition-all shadow-lg"
              >
                ⏹️ Stop Stream
              </button>
            )}
          </div>
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-20">
        <div className="container mx-auto px-6 py-8 text-center text-gray-500">
          <p className="text-base"><span className="font-bold">AIT102 Project</span> - Real-time Style Transfer</p>
          <p className="mt-2 text-sm">Powered by {isGodMode ? 'ONNX Runtime & DirectML' : 'Web Canvas API'}</p>
        </div>
      </footer>
    </div>
  )
}
