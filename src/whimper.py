"""
Whimper - GPU-Accelerated Live Audio Transcription with Whisper v3

A high-performance real-time speech-to-text transcription application using OpenAI's Whisper large-v3 model
with GPU acceleration via faster-whisper for maximum speed and accuracy.
"""

import logging
from typing import Optional, Callable
import queue
import threading
import time
import torch

try:
    import pyaudio
    import numpy as np
    from faster_whisper import WhisperModel
    import soundfile as sf
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioConfig:
    """Audio configuration constants optimized for Whisper"""
    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    CHANNELS = 1         # Mono audio
    CHUNK_SIZE = 1024    # Audio buffer size
    FORMAT = pyaudio.paInt16  # 16-bit audio
    BYTES_PER_SAMPLE = 2
    BUFFER_DURATION = 3.0  # Process audio every 3 seconds for faster response

class GPULiveTranscriber:
    """
    High-performance GPU-accelerated real-time audio transcription using Whisper large-v3.
    
    Features:
    - GPU acceleration with faster-whisper
    - Whisper large-v3 model for maximum accuracy
    - Optimized for real-time performance
    - Automatic device detection (CUDA/CPU)
    - Voice activity detection
    """
    
    def __init__(
        self, 
        model_size: str = "large-v3",
        model_path: Optional[str] = None,
        language: str = "en",
        device: str = "auto",
        compute_type: str = "auto",
        callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the GPU-accelerated live transcriber.
        
        Args:
            model_size: Whisper model size ('large-v3', 'large-v2', 'medium', etc.)
            model_path: Local path to model directory (optional, will download if not provided)
            language: Source language code ('en', 'auto', etc.)
            device: Device to use ('cuda', 'cpu', 'auto')
            compute_type: Compute type ('float16', 'int8', 'auto')
            callback: Optional callback function for transcription results
        """
        self.model_size = model_size
        self.model_path = model_path
        self.language = language
        self.callback = callback
        
        # Determine optimal device and compute type
        self.device, self.compute_type = self._determine_device_config(device, compute_type)
        
        # Audio components
        self.audio = None
        self.stream = None
        self.audio_queue = queue.Queue()
        
        # Whisper components
        self.model = None
        self.is_original_whisper = False
        
        # Control flags
        self.is_running = False
        self.is_recording = False
        
        # Audio buffer
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Performance tracking
        self.transcription_count = 0
        self.total_processing_time = 0.0
        
        # Initialize components
        self._initialize_audio()
        self._initialize_whisper()
    
    def _determine_device_config(self, device: str, compute_type: str):
        """Determine optimal device and compute type based on hardware"""
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        if device == "auto":
            if cuda_available:
                actual_device = "cuda"
                logger.info(f"🚀 CUDA detected! GPU: {torch.cuda.get_device_name()}")
            else:
                actual_device = "cpu"
                logger.info("💻 Using CPU (CUDA not available)")
        else:
            actual_device = device
            
        if compute_type == "auto":
            if actual_device == "cuda":
                # Use float16 for GPU for speed
                actual_compute_type = "float16"
            else:
                # Use int8 for CPU for efficiency
                actual_compute_type = "int8"
        else:
            actual_compute_type = compute_type
            
        logger.info(f"⚙️ Device: {actual_device}, Compute type: {actual_compute_type}")
        return actual_device, actual_compute_type
    
    def _initialize_audio(self):
        """Initialize PyAudio for microphone input"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # List available audio devices for debugging
            logger.info("🎤 Available audio input devices:")
            device_count = 0
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_count += 1
                    if device_count <= 5:  # Show only first 5 to avoid clutter
                        logger.info(f"   {i}: {info['name'][:50]}{'...' if len(info['name']) > 50 else ''}")
            
            if device_count > 5:
                logger.info(f"   ... and {device_count - 5} more devices")
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize audio: {e}")
            raise
    
    def _initialize_whisper(self):
        """Initialize faster-whisper model with GPU acceleration"""
        try:
            if self.model_path:
                logger.info(f"🤖 Loading local Whisper model from {self.model_path}...")
                model_source = self.model_path
                local_files_only = True
                
                # Try faster-whisper first, fallback to original whisper if needed
                try:
                    start_time = time.time()
                    self.model = WhisperModel(
                        model_source,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=None,
                        local_files_only=local_files_only,
                    )
                    logger.info("✅ Loaded with faster-whisper (CTranslate2 format)")
                    
                except Exception as e:
                    logger.warning(f"⚠️  faster-whisper failed: {e}")
                    logger.info("🔄 Trying original whisper format...")
                    
                    # Import original whisper as fallback
                    import whisper
                    start_time = time.time()
                    self.model = whisper.load_model(self.model_path)
                    self.is_original_whisper = True
                    logger.info("✅ Loaded with original whisper")
            else:
                logger.info(f"🤖 Loading Whisper {self.model_size} model...")
                logger.info(f"📦 This may take a moment to download (~1.5GB for large-v3)...")
                model_source = self.model_size
                local_files_only = False
                
                start_time = time.time()
                
                # Initialize faster-whisper model
                self.model = WhisperModel(
                    model_source,
                    device=self.device,
                    compute_type=self.compute_type,
                    # Enable optimizations
                    download_root=None,  # Use default cache
                    local_files_only=local_files_only,
                )
                self.is_original_whisper = False
            
            load_time = time.time() - start_time
            logger.info(f"✅ Whisper model loaded in {load_time:.1f}s")
            
            # Get model info
            logger.info(f"📊 Model parameters: ~{self._estimate_parameters()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper: {e}")
            raise
    
    def _estimate_parameters(self):
        """Estimate model parameters for display"""
        param_counts = {
            "large-v3": "1550M",
            "large-v2": "1550M", 
            "large": "1550M",
            "medium": "769M",
            "small": "244M",
            "base": "74M",
            "tiny": "39M"
        }
        return param_counts.get(self.model_size, "Unknown")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for capturing audio data"""
        if status:
            logger.warning(f"⚠️ Audio callback status: {status}")
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        if self.is_recording:
            with self.buffer_lock:
                self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def _processing_thread(self):
        """Background thread for processing audio chunks with GPU acceleration"""
        logger.info("🔄 GPU processing thread started")
        
        while self.is_running:
            try:
                # Wait for buffer to fill up
                time.sleep(AudioConfig.BUFFER_DURATION)
                
                if not self.is_recording:
                    continue
                
                # Get audio buffer
                with self.buffer_lock:
                    if len(self.audio_buffer) == 0:
                        continue
                    
                    # Convert to numpy array
                    audio_chunk = np.array(self.audio_buffer, dtype=np.float32)
                    
                    # Clear buffer for next chunk
                    self.audio_buffer = []
                
                # Skip if chunk is too short
                if len(audio_chunk) < AudioConfig.SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
                    logger.info(f"🔇 Skipping short audio chunk ({len(audio_chunk)/AudioConfig.SAMPLE_RATE:.1f}s)")
                    continue
                
                # Check audio level
                audio_level = float(np.max(np.abs(audio_chunk)))
                logger.info(f"🎵 Audio level: {audio_level:.3f} (duration: {len(audio_chunk)/AudioConfig.SAMPLE_RATE:.1f}s)")
                
                # Process with model (GPU accelerated or original whisper)
                start_time = time.time()
                
                if self.is_original_whisper:
                    logger.info("🚀 Processing audio chunk with original Whisper...")
                    # Original whisper transcription
                    result = self.model.transcribe(
                        audio_chunk,
                        language=None if self.language == "auto" else self.language,
                        verbose=False
                    )
                    text = result["text"].strip()
                else:
                    logger.info("🚀 Processing audio chunk with GPU acceleration...")
                    
                    # Transcribe with faster-whisper (testing mode - process even silence)
                    if audio_level < 0.001:  # Very quiet audio
                        logger.info("🔇 Audio level too low for transcription, skipping...")
                        text = ""  # Skip transcription for silent audio
                    else:
                        segments, info = self.model.transcribe(
                            audio_chunk,
                            language=None if self.language == "auto" else self.language,
                            vad_filter=True,  # Enable voice activity detection
                            vad_parameters=dict(
                                min_silence_duration_ms=300,  # Reduced sensitivity
                                threshold=0.3,                # Lower threshold
                            ),
                            beam_size=3,  # Faster processing
                            temperature=0.0,  # Deterministic output
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                            no_speech_threshold=0.4,  # Reduced from 0.6
                            condition_on_previous_text=False,  # Each chunk is independent
                        )
                        
                        # Collect transcription text
                        text_segments = []
                        for segment in segments:
                            text_segments.append(segment.text.strip())
                        
                        text = " ".join(text_segments).strip()
                
                processing_time = time.time() - start_time
                self.transcription_count += 1
                self.total_processing_time += processing_time
                
                if text:
                    # Performance info
                    avg_time = self.total_processing_time / self.transcription_count
                    logger.info(f"⚡ Processed in {processing_time:.2f}s (avg: {avg_time:.2f}s)")
                    
                    # Call callback if provided
                    if self.callback:
                        self.callback(text)
                    else:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"🎯 [{timestamp}] {text}")
                        
            except Exception as e:
                logger.error(f"❌ Error in processing thread: {e}")
    
    def start_recording(self, device_index: Optional[int] = None):
        """
        Start GPU-accelerated live audio recording and transcription.
        
        Args:
            device_index: Optional audio device index. If None, uses default.
        """
        if self.is_recording:
            logger.warning("⚠️ Recording already in progress")
            return
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=AudioConfig.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            # Start flags
            self.is_running = True
            self.is_recording = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start audio stream
            self.stream.start_stream()
            
            logger.info("🎤 GPU-accelerated live transcription started!")
            
        except Exception as e:
            logger.error(f"❌ Failed to start recording: {e}")
            self.stop_recording()
            raise
    
    def stop_recording(self):
        """Stop live audio recording and transcription"""
        if not self.is_recording:
            return
        
        logger.info("🛑 Stopping live transcription...")
        
        # Stop recording
        self.is_recording = False
        self.is_running = False
        
        # Stop and close audio stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"⚠️ Error stopping audio stream: {e}")
        
        # Performance summary
        if self.transcription_count > 0:
            avg_time = self.total_processing_time / self.transcription_count
            logger.info(f"📊 Performance: {self.transcription_count} chunks, avg {avg_time:.2f}s per chunk")
        
        logger.info("✅ Live transcription stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"⚠️ Error terminating PyAudio: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

def gpu_transcription_callback(text: str):
    """Enhanced callback function for handling GPU transcription results"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"🚀🎤 [{timestamp}] {text}")

def main():
    """Main function for GPU-accelerated command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whimper - GPU-Accelerated Live Audio Transcription")
    parser.add_argument("--model", default="large-v3", 
                       choices=["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
                       help="Whisper model size (default: large-v3 for best accuracy)")
    parser.add_argument("--model-path", help="Path to local Whisper model directory")
    parser.add_argument("--language", default="en", help="Source language code")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use (auto=detect GPU)")
    parser.add_argument("--compute-type", default="auto", 
                       choices=["auto", "float16", "int8", "float32"],
                       help="Compute precision (auto=optimal for device)")
    parser.add_argument("--audio-device", type=int, help="Audio device index")
    
    args = parser.parse_args()
    
    print("🚀 Whimper - GPU-Accelerated Live Audio Transcription")
    print("=" * 55)
    print(f"🤖 Model: {args.model}")
    if args.model_path:
        print(f"📁 Model Path: {args.model_path}")
    print(f"🌍 Language: {args.language}")
    print(f"⚙️  Device: {args.device}")
    print(f"🔧 Compute: {args.compute_type}")
    print(f"🎤 Audio Device: {args.audio_device if args.audio_device is not None else 'Default'}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Create GPU-accelerated transcriber
        with GPULiveTranscriber(
            model_size=args.model,
            model_path=args.model_path,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            callback=gpu_transcription_callback
        ) as transcriber:
            
            # Start transcription
            transcriber.start_recording(device_index=args.audio_device)
            
            print("🚀 GPU-accelerated listening active! Speak into your microphone!")
            print(f"⚡ Audio processed every {AudioConfig.BUFFER_DURATION} seconds with GPU acceleration\n")
            
            # Keep running until user interrupts
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping transcription...")
                
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())