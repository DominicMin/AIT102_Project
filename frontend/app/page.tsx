'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

interface StyleResult {
  style_id: string
  image_base64: string
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<StyleResult[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [isVideo, setIsVideo] = useState(false)
  const [showVideoModal, setShowVideoModal] = useState(false)
  const [selectedStyle, setSelectedStyle] = useState<string>('sketch')
  const [videoProcessing, setVideoProcessing] = useState(false)
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleFile = (file: File) => {
    const isVideoFile = file.type.startsWith('video/')
    const isImageFile = file.type.startsWith('image/')
    
    if (isVideoFile || isImageFile) {
      setSelectedFile(file)
      setIsVideo(isVideoFile)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string)
      }
      reader.readAsDataURL(file)
      setResults([])
      setProcessedVideoUrl(null)
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleTransformAll = async () => {
    if (!selectedFile) return

    setLoading(true)
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post('/api/transform_all', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setResults(response.data.results)
    } catch (error) {
      console.error('Transform failed:', error)
      alert('Failed to transform image. Please ensure the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleTransformVideo = async () => {
    if (!selectedFile || !selectedStyle) return

    setVideoProcessing(true)
    setShowVideoModal(false)
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post(`/api/transform_video?style_id=${selectedStyle}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
        timeout: 1800000, // 30 minutes
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        onUploadProgress: (progressEvent) => {
          console.log('Upload progress:', progressEvent.loaded, '/', progressEvent.total)
        },
      })
      
      const videoBlob = new Blob([response.data], { type: 'video/mp4' })
      const videoUrl = URL.createObjectURL(videoBlob)
      setProcessedVideoUrl(videoUrl)
      
      alert('Video processed successfully!')
    } catch (error: any) {
      console.error('Video transform failed:', error)
      if (error.code === 'ECONNABORTED') {
        alert('Video processing timeout. The video might be too large or complex.')
      } else if (error.code === 'ECONNRESET') {
        alert('Connection was reset. This might be due to server timeout. Try with a shorter video.')
      } else if (error.response) {
        alert(`Failed to transform video: ${error.response.status} - ${error.response.statusText}`)
      } else {
        alert(`Failed to transform video: ${error.message || 'Unknown error'}. Please ensure the backend server is running.`)
      }
    } finally {
      setVideoProcessing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 backdrop-blur-sm bg-black/30">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-600 bg-clip-text text-transparent">
              AI Style Transfer
            </h1>
            <div className="flex items-center gap-4 text-base text-gray-400">
              <span className="font-bold">AIT102 Project</span>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-12">
        {/* Hero Section */}
        <AnimatePresence mode="wait">
          {!results.length ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-3xl mx-auto text-center"
            >
              <h2 className="text-5xl md:text-7xl font-bold mb-6 animate-gradient bg-gradient-to-r from-blue-400 via-cyan-500 to-blue-600 bg-clip-text text-transparent">
                Transform Your Art
              </h2>
              <p className="text-xl text-gray-400 mb-12">
                Upload an image or video and watch AI transform it into multiple artistic styles
              </p>

              {/* Upload Area */}
              <div className="space-y-6">
                <div
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  className={`relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 ${
                    dragActive
                      ? 'border-blue-500 bg-blue-500/10'
                      : previewUrl
                      ? 'border-gray-700'
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  {!previewUrl && (
                    <input
                      type="file"
                      onChange={handleFileInput}
                      accept="image/*,video/*"
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                  )}
                  
                  {previewUrl ? (
                    <div className="space-y-4">
                      {isVideo ? (
                        <video
                          key={previewUrl}
                          src={previewUrl}
                          controls
                          controlsList="nodownload"
                          preload="auto"
                          playsInline
                          muted
                          className="max-h-64 mx-auto rounded-lg shadow-2xl"
                          style={{ maxWidth: '100%', display: 'block' }}
                          onError={(e) => {
                            console.error('Video preview error:', e)
                            const target = e.target as HTMLVideoElement
                            console.error('Video error details:', {
                              error: target.error,
                              networkState: target.networkState,
                              readyState: target.readyState
                            })
                          }}
                          onLoadedMetadata={(e) => {
                            console.log('Video metadata loaded')
                            const target = e.target as HTMLVideoElement
                            console.log('Video details:', {
                              duration: target.duration,
                              videoWidth: target.videoWidth,
                              videoHeight: target.videoHeight
                            })
                          }}
                          onCanPlay={() => console.log('Video can play')}
                        />
                      ) : (
                        <img
                          src={previewUrl}
                          alt="Preview"
                          className="max-h-64 mx-auto rounded-lg shadow-2xl"
                        />
                      )}
                      <label className="inline-block px-6 py-2 bg-gray-800 hover:bg-gray-700 rounded-full text-sm cursor-pointer transition-colors">
                        <input
                          type="file"
                          onChange={handleFileInput}
                          accept="image/*,video/*"
                          className="hidden"
                        />
                        Change {isVideo ? 'Video' : 'Image'}
                      </label>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="w-20 h-20 mx-auto bg-gray-800 rounded-full flex items-center justify-center">
                        <svg
                          className="w-10 h-10 text-gray-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                          />
                        </svg>
                      </div>
                      <div>
                        <p className="text-xl font-semibold mb-2">Drop your image or video here</p>
                        <p className="text-gray-500">or click to browse</p>
                      </div>
                    </div>
                  )}
                </div>

                {previewUrl && (
                  <div className="space-y-4">
                    {videoProcessing && (
                      <div className="p-6 bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-xl border border-purple-500/30">
                        <div className="flex items-center justify-center gap-3 mb-2">
                          <svg className="animate-spin h-6 w-6 text-purple-400" viewBox="0 0 24 24">
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                              fill="none"
                            />
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            />
                          </svg>
                          <span className="text-lg font-semibold text-purple-200">Processing Video...</span>
                        </div>
                        <p className="text-sm text-purple-300 text-center">This may take several minutes depending on video length</p>
                      </div>
                    )}
                    
                    {!isVideo ? (
                      <button
                        onClick={handleTransformAll}
                        disabled={loading}
                        className="w-full px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-full font-semibold text-lg hover:from-blue-700 hover:to-cyan-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
                      >
                        {loading ? (
                          <span className="flex items-center justify-center gap-2">
                            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                              <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                                fill="none"
                              />
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                              />
                            </svg>
                            Processing...
                          </span>
                        ) : (
                          '✨ Transform with AI'
                        )}
                      </button>
                    ) : (
                      <button
                        onClick={() => setShowVideoModal(true)}
                        disabled={videoProcessing}
                        className="w-full px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full font-semibold text-lg hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
                      >
                        🎬 Process Video with Style
                      </button>
                    )}
                    
                    {processedVideoUrl && isVideo && (
                      <div className="mt-6 p-6 bg-gray-800 rounded-xl">
                        <h3 className="text-xl font-bold mb-4">Processed Video</h3>
                        <video
                          key={processedVideoUrl}
                          src={processedVideoUrl}
                          controls
                          controlsList="nodownload"
                          preload="auto"
                          playsInline
                          muted
                          className="w-full rounded-lg shadow-2xl mb-4"
                          style={{ display: 'block' }}
                          onError={(e) => {
                            console.error('Processed video error:', e)
                            const target = e.target as HTMLVideoElement
                            console.error('Processed video error details:', {
                              error: target.error,
                              networkState: target.networkState,
                              readyState: target.readyState
                            })
                          }}
                          onLoadedMetadata={(e) => {
                            console.log('Processed video metadata loaded')
                            const target = e.target as HTMLVideoElement
                            console.log('Processed video details:', {
                              duration: target.duration,
                              videoWidth: target.videoWidth,
                              videoHeight: target.videoHeight
                            })
                          }}
                        />
                        <a
                          href={processedVideoUrl}
                          download={`styled_video_${selectedStyle}.mp4`}
                          className="inline-block px-6 py-3 bg-green-600 hover:bg-green-700 rounded-full font-semibold transition-colors"
                        >
                          📥 Download Video
                        </a>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-8"
            >
              <div className="text-center">
                <button
                  onClick={() => {
                    setResults([])
                    setSelectedFile(null)
                    setPreviewUrl(null)
                  }}
                  className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors text-xl font-bold"
                >
                  <svg 
                    className="w-6 h-6" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                    strokeWidth={2.5}
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      d="M10 19l-7-7m0 0l7-7m-7 7h18"
                    />
                  </svg>
                  Try another image
                </button>
              </div>

              {/* Results Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Original Image */}
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="relative group"
                >
                  <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg blur opacity-25 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative bg-gray-900 rounded-lg overflow-hidden shadow-xl">
                    <img
                      src={previewUrl || ''}
                      alt="Original"
                      className="w-full h-64 object-cover"
                    />
                    <div className="p-4 border-t border-gray-800">
                      <h3 className="font-semibold text-lg">Original</h3>
                    </div>
                  </div>
                </motion.div>

                {/* Styled Results */}
                {results.map((result, index) => (
                  <motion.div
                    key={result.style_id}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: index * 0.1 }}
                    className="relative group"
                  >
                    <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg blur opacity-25 group-hover:opacity-100 transition duration-300"></div>
                    <div className="relative bg-gray-900 rounded-lg overflow-hidden shadow-xl">
                      <img
                        src={`data:image/jpeg;base64,${result.image_base64}`}
                        alt={result.style_id}
                        className="w-full h-64 object-cover"
                      />
                      <div className="p-4 border-t border-gray-800">
                        <h3 className="font-semibold text-lg capitalize">
                          {result.style_id.replace('_', ' ')}
                        </h3>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Video Style Selection Modal */}
      {showVideoModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-6">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-gray-900 rounded-2xl p-8 max-w-md w-full border border-gray-700"
          >
            <h3 className="text-2xl font-bold mb-6">Select Video Style</h3>
            
            <div className="space-y-4 mb-6">
              {['sketch', 'cyberpunk', 'picasso', 'vangogh'].map((style) => (
                <button
                  key={style}
                  onClick={() => setSelectedStyle(style)}
                  className={`w-full px-6 py-4 rounded-xl font-semibold text-lg transition-all ${
                    selectedStyle === style
                      ? 'bg-gradient-to-r from-blue-600 to-cyan-600 shadow-lg'
                      : 'bg-gray-800 hover:bg-gray-700'
                  }`}
                >
                  {style.charAt(0).toUpperCase() + style.slice(1)}
                </button>
              ))}
            </div>

            <div className="flex gap-4">
              <button
                onClick={() => setShowVideoModal(false)}
                className="flex-1 px-6 py-3 bg-gray-800 hover:bg-gray-700 rounded-full font-semibold transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleTransformVideo}
                disabled={videoProcessing}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full font-semibold hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50"
              >
                {videoProcessing ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Processing...
                  </span>
                ) : (
                  'Process'
                )}
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-20">
        <div className="container mx-auto px-6 py-8 text-center text-gray-500">
          <p className="text-base"><span className="font-bold">AIT102 Project</span> - High-Performance Style Transfer System</p>
          <p className="mt-2 text-sm">Powered by TensorFlow & Next.js</p>
        </div>
      </footer>
    </div>
  )
}
