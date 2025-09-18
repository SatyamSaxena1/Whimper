"""
Example: Advanced live transcription with custom settings

This example shows how to use Whimper with custom configurations,
device selection, and advanced features.
"""

import sys
import os
import time
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from whimper import GPULiveTranscriber
import pyaudio

class TranscriptionLogger:
    """Advanced callback class that logs transcriptions to file"""
    
    def __init__(self, output_file: str = "transcription.json"):
        self.output_file = output_file
        self.transcriptions = []
        
    def __call__(self, text: str):
        """Called for each transcription result"""
        entry = {
            "timestamp": time.time(),
            "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "is_final": text.startswith("[FINAL]")
        }
        
        self.transcriptions.append(entry)
        
        # Print with nice formatting
        prefix = "📝" if entry["is_final"] else "🔊"
        print(f"{prefix} [{entry['formatted_time']}] {text}")
        
        # Save to file periodically
        if len(self.transcriptions) % 10 == 0:
            self.save_to_file()
    
    def save_to_file(self):
        """Save transcriptions to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.transcriptions, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(self.transcriptions)} transcriptions to {self.output_file}")
        except Exception as e:
            print(f"❌ Error saving to file: {e}")

def list_audio_devices():
    """List all available audio input devices"""
    audio = pyaudio.PyAudio()
    
    print("\n🎤 Available audio input devices:")
    print("-" * 50)
    
    devices = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append((i, info))
            print(f"  {i}: {info['name']}")
            print(f"      Max inputs: {info['maxInputChannels']}")
            print(f"      Sample rate: {info['defaultSampleRate']}")
            print()
    
    audio.terminate()
    return devices

def main():
    print("Whimper Advanced Live Transcription Example")
    print("==========================================")
    
    # List available devices
    devices = list_audio_devices()
    
    if not devices:
        print("❌ No audio input devices found!")
        return 1
    
    # Get user device selection
    device_index = None
    if len(devices) > 1:
        try:
            choice = input(f"Select device (0-{len(devices)-1}, or Enter for default): ")
            if choice.strip():
                device_index = int(choice)
                if device_index < 0 or device_index >= len(devices):
                    print("❌ Invalid device selection, using default")
                    device_index = None
        except ValueError:
            print("❌ Invalid input, using default device")
            device_index = None
    
    # Get model selection
    models = ["turbo", "large-v3", "medium", "small", "base"]
    print(f"\n🤖 Available models: {', '.join(models)}")
    model_choice = input("Select model (or Enter for 'turbo'): ").strip()
    if model_choice not in models:
        model_choice = "turbo"
    
    # Get language selection
    languages = ["en", "es", "fr", "de", "it", "pt", "auto"]
    print(f"\n🌍 Languages: {', '.join(languages)}")
    lang_choice = input("Select language (or Enter for 'en'): ").strip()
    if lang_choice not in languages:
        lang_choice = "en"
    
    # Create logger
    logger = TranscriptionLogger("advanced_transcription.json")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: {model_choice}")
    print(f"   Language: {lang_choice}")
    print(f"   Device: {device_index if device_index is not None else 'Default'}")
    print(f"   Output: advanced_transcription.json")
    print(f"\nPress Ctrl+C to stop\n")
    
    try:
        # Create transcriber with custom settings
        with GPULiveTranscriber(
            model_size=model_choice,
            language=lang_choice,
            callback=logger
        ) as transcriber:
            
            # Start live transcription
            transcriber.start_recording(device_index=device_index)
            
            print("🎤 Advanced transcription started! Speak into your microphone!")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n📝 Stopping transcription...")
                
        # Save final results
        logger.save_to_file()
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("✅ Advanced transcription complete!")
    return 0

if __name__ == "__main__":
    exit(main())