# Whimper 🎤

**Live Audio Transcription with Whisper v3**

A real-time speech-to-text transcription application using OpenAI's Whisper v3 model. Whimper provides streaming transcription with word-level timestamps, voice activity detection, and cross-platform audio support.

## ✨ Features

- **Real-time Transcription**: Live audio processing with minimal latency
- **Whisper v3 Support**: Uses the latest Whisper models including the optimized "turbo" variant
- **Voice Activity Detection**: Intelligent audio processing with VAD
- **Multi-language Support**: Supports 98+ languages with automatic detection
- **Cross-platform Audio**: Works on Windows, macOS, and Linux
- **Streaming Processing**: Continuous transcription with immediate results
- **Customizable**: Flexible configuration for different use cases
- **Python API**: Easy integration into existing applications

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SatyamSaxena1/Whimper.git
cd whimper
```

2. **Install system dependencies**:

**Windows**:
```bash
# PyAudio is pre-compiled for Windows
# No additional system dependencies needed
```

**macOS**:
```bash
# Install PortAudio via Homebrew
brew install portaudio
```

**Linux (Ubuntu/Debian)**:
```bash
# Install system audio libraries
sudo apt update && sudo apt install -y \
    python3-pyaudio \
    portaudio19-dev \
    ffmpeg
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

### Basic Usage

**Command Line**:
```bash
# Simple transcription with default settings
python main.py

# With custom model and language
python main.py --model turbo --language en --device 0
```

**Python API**:
```python
from src.whimper import GPULiveTranscriber

def on_transcription(text):
    print(f"Transcribed: {text}")

# Create and start transcriber
with GPULiveTranscriber(
    model_size="turbo",
    language="en", 
    callback=on_transcription
) as transcriber:
    transcriber.start_recording()
    
    # Keep running...
    input("Press Enter to stop...")
```

## 📚 Examples

### Basic Live Transcription
```bash
python examples/basic_usage.py
```
Demonstrates simple real-time transcription with default settings.

### Advanced Configuration
```bash
python examples/advanced_usage.py
```
Shows device selection, model configuration, and output logging.

## 🔧 Configuration

### Model Options
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `turbo` | 798M | ⚡⚡⚡ | ⭐⭐⭐ | **Real-time (Recommended)** |
| `large-v3` | 1550M | ⚡⚡ | ⭐⭐⭐⭐ | High accuracy |
| `medium` | 769M | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `small` | 244M | ⚡⚡⚡ | ⭐⭐ | Fast processing |
| `base` | 74M | ⚡⚡⚡ | ⭐ | Minimal resources |

### Language Support
- **Auto-detection**: Use `"auto"` for automatic language detection
- **Specific languages**: `"en"`, `"es"`, `"fr"`, `"de"`, `"zh"`, etc.
- **Translation**: Set task to `"translate"` for real-time translation to English

### Audio Configuration
```python
# Default audio settings (optimal for Whisper)
SAMPLE_RATE = 16000  # 16kHz (required by Whisper)
CHANNELS = 1         # Mono audio
CHUNK_SIZE = 1024    # Audio buffer size
FORMAT = pyaudio.paInt16  # 16-bit audio
```

## 🔍 API Reference

### GPULiveTranscriber Class

```python
class GPULiveTranscriber:
    def __init__(
        self, 
        model_size: str = "large-v3",
        language: str = "en",
        device: str = "auto",
        compute_type: str = "auto",
        callback: Optional[Callable[[str], None]] = None
    )
```

**Parameters**:
- `model_size`: Whisper model size (`"turbo"`, `"large-v3"`, etc.)
- `language`: Source language code (`"en"`, `"auto"`, etc.)
- `device`: Device to use (`"cuda"`, `"cpu"`, `"auto"`)
- `compute_type`: Compute precision (`"float16"`, `"int8"`, `"auto"`)
- `callback`: Function called with transcription results

**Methods**:
- `start_recording(device_index=None)`: Start live transcription
- `stop_recording()`: Stop transcription and finalize results
- `cleanup()`: Release resources

## 🎛️ Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --model TEXT        Whisper model size [default: large-v3]
  --language TEXT     Source language code [default: en]  
  --device TEXT       Device to use [default: auto]
  --compute-type TEXT Compute precision [default: auto]
  --audio-device INT  Audio device index [default: auto]
  --help             Show help message
```

## 🔧 Troubleshooting

### Common Issues

**"Import pyaudio could not be resolved"**:
- **Windows**: `pip install pyaudio`
- **macOS**: `brew install portaudio && pip install pyaudio`
- **Linux**: `sudo apt install python3-pyaudio`

**"No audio devices found"**:
- Check microphone permissions
- Verify audio device connection
- List devices: `python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels']>0]"`

**"Whisper model download failed"**:
- Check internet connection
- Verify disk space (models are 100MB-1.5GB)
- Try smaller model size first

**Poor transcription quality**:
- Ensure good microphone quality
- Reduce background noise
- Check language setting
- Try larger model (`large-v3`)
- Enable VAD for noise filtering

**High CPU usage**:
- Use smaller model (`turbo` or `small`)
- Increase audio chunk processing time
- Enable VAD to reduce processing

### Performance Optimization

**For Real-time Performance**:
- Use `turbo` model (optimized for speed)
- Enable VAD for noise filtering
- Use GPU if available (automatic with `faster-whisper`)
- Set specific language instead of auto-detection

**For Accuracy**:
- Use `large-v3` model
- Disable background noise
- Use good quality microphone
- Set specific language instead of auto-detection

## 📦 Dependencies

### Core Dependencies
- `openai-whisper`: Official Whisper implementation
- `faster-whisper`: Optimized Whisper backend (4x faster)
- `pyaudio`: Cross-platform audio I/O
- `librosa`: Audio processing utilities
- `soundfile`: Audio file I/O
- `numpy`: Numerical operations

### Optional Dependencies
- `torch` + `torchaudio`: For VAD and advanced features
- `opus-fast-mosestokenizer`: Improved sentence segmentation
- `mlx-whisper`: Apple Silicon optimization (M1/M2 only)

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test:
```bash
python tests/test_whimper.py
```

## 🏗️ Development

### Project Structure
```
whimper/
├── src/
│   └── whimper.py          # Main transcription module
├── examples/
│   ├── basic_usage.py      # Simple example
│   └── advanced_usage.py   # Advanced configuration
├── tests/
│   └── test_whimper.py     # Test suite
├── .github/
│   └── copilot-instructions.md  # Project documentation
├── main.py                 # Command-line interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Ensure all tests pass: `python -m pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for the Whisper model
- **faster-whisper** contributors for performance optimizations
- **PyAudio** team for cross-platform audio support

## 🔗 Related Projects

- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper implementation
- [faster-whisper](https://github.com/systran/faster-whisper) - Optimized inference
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) - Python audio I/O

---

**Made with ❤️ for real-time speech recognition**