# Whimper - Live Audio Transcription Project

## Project Overview
This is a Python project that provides real-time speech-to-text transcription using OpenAI's Whisper v3 model. The application captures live audio from the microphone and provides streaming transcription with word-level timestamps.

## Key Technologies
- **OpenAI Whisper v3**: Latest speech recognition model with "turbo" variant for real-time performance
- **whisper-streaming**: Real-time streaming capabilities from ÚFAL
- **faster-whisper**: Optimized backend for 4x faster processing
- **PyAudio**: Cross-platform real-time audio capture
- **PyTorch**: For voice activity detection and advanced features

## Project Structure
- `src/whimper.py`: Main transcription module with LiveTranscriber class
- `main.py`: Command-line interface
- `examples/`: Basic and advanced usage examples
- `tests/`: Comprehensive test suite
- `requirements.txt`: All dependencies for the live transcription system

## Development Guidelines
- Use the "turbo" model for real-time applications (798M parameters, optimized for speed)
- Enable VAD (Voice Activity Detection) for better performance
- Support multiple languages with auto-detection capabilities
- Maintain cross-platform compatibility (Windows, macOS, Linux)
- Follow real-time processing patterns with streaming audio chunks

## Installation & Setup
1. Install system audio dependencies (portaudio, ffmpeg)
2. Install Python requirements: `pip install -r requirements.txt`
3. Run basic example: `python examples/basic_usage.py`
4. Or use CLI: `python main.py --model turbo --language en`

## Key Features Implemented
- Real-time microphone input processing
- Streaming transcription with immediate results
- Voice activity detection integration
- Multi-device audio support
- Configurable models and languages
- Python API and CLI interfaces
- Comprehensive error handling and logging