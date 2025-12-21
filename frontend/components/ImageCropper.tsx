import { useState, useCallback } from 'react'
import Cropper from 'react-easy-crop'
import { motion, AnimatePresence } from 'framer-motion'
import getCroppedImg from '../utils/cropImage'

interface ImageCropperProps {
    imageSrc: string
    onCropComplete: (croppedBlob: Blob) => void
    onCancel: () => void
}

export default function ImageCropper({ imageSrc, onCropComplete, onCancel }: ImageCropperProps) {
    const [crop, setCrop] = useState({ x: 0, y: 0 })
    const [zoom, setZoom] = useState(1)
    const [croppedAreaPixels, setCroppedAreaPixels] = useState<any>(null)

    const onCropChange = (crop: { x: number; y: number }) => {
        setCrop(crop)
    }

    const onZoomChange = (zoom: number) => {
        setZoom(zoom)
    }

    const onCropCompleteHandler = useCallback((croppedArea: any, croppedAreaPixels: any) => {
        setCroppedAreaPixels(croppedAreaPixels)
    }, [])

    const handleSave = async () => {
        try {
            const croppedImage = await getCroppedImg(imageSrc, croppedAreaPixels)
            if (croppedImage) {
                onCropComplete(croppedImage)
            }
        } catch (e) {
            console.error(e)
        }
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4">
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="relative w-full max-w-2xl bg-gray-900 rounded-2xl overflow-hidden shadow-2xl border border-gray-700 flex flex-col h-[80vh]"
            >
                <div className="p-4 border-b border-gray-800 flex justify-between items-center">
                    <h3 className="text-xl font-bold text-white">Crop Image</h3>
                    <button onClick={onCancel} className="text-gray-400 hover:text-white">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <div className="relative flex-1 bg-black">
                    <Cropper
                        image={imageSrc}
                        crop={crop}
                        zoom={zoom}
                        aspect={1}
                        onCropChange={onCropChange}
                        onCropComplete={onCropCompleteHandler}
                        onZoomChange={onZoomChange}
                    />
                </div>

                <div className="p-6 bg-gray-900 border-t border-gray-800 space-y-4">
                    <div className="flex items-center gap-4">
                        <span className="text-sm font-medium text-gray-300">Zoom</span>
                        <input
                            type="range"
                            value={zoom}
                            min={1}
                            max={3}
                            step={0.1}
                            aria-labelledby="Zoom"
                            onChange={(e) => setZoom(Number(e.target.value))}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                    </div>

                    <div className="flex gap-4">
                        <button
                            onClick={onCancel}
                            className="flex-1 px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-xl font-semibold transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            className="flex-1 px-4 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl font-semibold hover:from-blue-700 hover:to-cyan-700 transition-all shadow-lg"
                        >
                            Crop & Continue
                        </button>
                    </div>
                </div>
            </motion.div>
        </div>
    )
}
