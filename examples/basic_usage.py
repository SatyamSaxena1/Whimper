"""
Example: Basic live transcription with Whimper

This example demonstrates the simplest usage of Whimper for live audio transcription.
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from whimper import GPULiveTranscriber, TranscriptionResult


def simple_callback(result: TranscriptionResult | str) -> None:
    """Simple callback that prints transcription with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    if isinstance(result, TranscriptionResult):
        status = "FINAL" if result.is_final else "LIVE"
        text = result.text
    else:
        status = "FINAL"
        text = str(result)
    print(f"[{timestamp}] {status} {text}")

def main():
    print("Whimper Live Transcription Example")
    print("==================================")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Create transcriber with default settings
        with GPULiveTranscriber(
            model_size="turbo",  # Fast model for real-time use
            language="en",       # English language
            callback=simple_callback
        ) as transcriber:
            
            # Start live transcription
            transcriber.start_recording()
            
            print("🎤 Listening... Speak into your microphone!")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n📝 Stopping transcription...")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("✅ Transcription complete!")
    return 0

if __name__ == "__main__":
    exit(main())