"""
Test suite for Whimper live transcription functionality.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import types
from unittest.mock import Mock, patch

try:  # pragma: no cover - allow running without numpy installed
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# Add src directory to path for imports
sys.path.insert(0, os.path.join(Path(__file__).resolve().parent, '..', 'src'))

# Provide a lightweight torch stub if the real package is unavailable.
if 'torch' not in sys.modules:  # pragma: no cover - testing fallback
    mock_cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda *args, **kwargs: "Mock GPU",
    )
    sys.modules['torch'] = types.SimpleNamespace(cuda=mock_cuda)

if 'pyaudio' not in sys.modules:  # pragma: no cover - testing fallback
    class _MockStream:
        def start_stream(self):
            return None

        def stop_stream(self):
            return None

        def close(self):
            return None

    class _MockPyAudio:
        def __init__(self, *args, **kwargs):
            pass

        def get_device_count(self):
            return 0

        def get_device_info_by_index(self, index):
            return {'name': 'Mock Device', 'maxInputChannels': 0}

        def open(self, *args, **kwargs):
            return _MockStream()

        def terminate(self):
            return None

    sys.modules['pyaudio'] = types.SimpleNamespace(PyAudio=_MockPyAudio, paInt16=8)

if 'faster_whisper' not in sys.modules:  # pragma: no cover - testing fallback
    class _MockWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            info = SimpleNamespace(language=None, language_probability=0.0)
            return [], info

    sys.modules['faster_whisper'] = types.SimpleNamespace(WhisperModel=_MockWhisperModel)


class TestAudioConfig(unittest.TestCase):
    """Test audio configuration constants."""

    def test_audio_config_values(self) -> None:
        from whimper import AudioConfig

        self.assertEqual(AudioConfig.SAMPLE_RATE, 16000)
        self.assertEqual(AudioConfig.CHANNELS, 1)
        self.assertEqual(AudioConfig.CHUNK_SIZE, 1024)
        self.assertEqual(AudioConfig.BYTES_PER_SAMPLE, 2)


class TestLiveTranscriber(unittest.TestCase):
    """Test GPULiveTranscriber class functionality."""

    def setUp(self) -> None:
        self.mock_audio = Mock()
        self.mock_model = Mock()

    @patch('whimper.pyaudio.PyAudio')
    @patch('whimper.WhisperModel')
    @patch('whimper.torch.cuda.is_available')
    def test_init_success(self, mock_cuda, mock_model_class, mock_audio_class):
        """Ensure the transcriber initialises audio and model resources."""
        mock_cuda.return_value = False
        mock_audio_class.return_value = self.mock_audio
        mock_model_class.return_value = self.mock_model

        self.mock_audio.get_device_count.return_value = 1
        self.mock_audio.get_device_info_by_index.return_value = {
            'name': 'Device 1',
            'maxInputChannels': 1,
        }

        from whimper import GPULiveTranscriber

        transcriber = GPULiveTranscriber(model_size="base", language="en")
        mock_model_class.assert_called_once()
        self.assertEqual(transcriber.model_size, "base")
        self.assertIsNotNone(transcriber.session)

        transcriber.cleanup()

    @patch('whimper.pyaudio.PyAudio')
    @patch('whimper.WhisperModel')
    @patch('whimper.torch.cuda.is_available')
    def test_callback_reference_stored(self, mock_cuda, mock_model_class, mock_audio_class):
        """The provided callback should be stored for later use."""
        mock_cuda.return_value = False
        mock_audio_class.return_value = self.mock_audio
        mock_model_class.return_value = self.mock_model

        self.mock_audio.get_device_count.return_value = 1
        self.mock_audio.get_device_info_by_index.return_value = {
            'name': 'Test Device',
            'maxInputChannels': 1,
        }

        from whimper import GPULiveTranscriber

        callback = Mock()
        transcriber = GPULiveTranscriber(callback=callback)
        self.assertIs(transcriber.callback, callback)
        transcriber.cleanup()

    @unittest.skipIf(np is None, "numpy not available")
    def test_audio_data_conversion(self):
        """Audio data is converted from int16 to float32 correctly."""
        samples = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
        expected = samples.astype(np.float32) / 32768.0
        converted = np.frombuffer(samples.tobytes(), dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(converted, expected)


class TestTranscriptionCallback(unittest.TestCase):
    """Test the default GPU transcription callback formatting."""

    @patch('builtins.print')
    @patch('time.strftime', return_value="12:34:56")
    def test_gpu_transcription_callback(self, mock_strftime, mock_print):
        from whimper import gpu_transcription_callback

        gpu_transcription_callback("Hello world")
        mock_print.assert_called_once_with("🚀🎤 [12:34:56] FINAL Hello world")


class TestStreamingSession(unittest.TestCase):
    """Tests for the streaming transcription session and helpers."""

    @unittest.skipIf(np is None, "numpy not available")
    def test_streaming_session_emits_results(self):
        from whimper import StreamingTranscriptionSession, AudioConfig

        mock_model = Mock()
        segments = [
            SimpleNamespace(text="hello", start=0.0, end=1.0, no_speech_prob=0.1),
            SimpleNamespace(text="hello world", start=1.0, end=2.0, no_speech_prob=0.1),
        ]
        info = SimpleNamespace(language="en", language_probability=0.9)
        mock_model.transcribe.return_value = (segments, info)

        session = StreamingTranscriptionSession(mock_model, language="auto", use_vad=False)
        audio = np.ones(AudioConfig.SAMPLE_RATE * 2, dtype=np.float32) * 0.05
        session.add_audio(audio)

        results = session.process_next()
        self.assertTrue(results)
        finals = [res for res in results if res.is_final]
        self.assertTrue(finals)
        self.assertEqual(session.detected_language, "en")

    @unittest.skipIf(np is None, "numpy not available")
    def test_simple_vad_detection(self):
        from whimper import SimpleVAD, AudioConfig

        vad = SimpleVAD(AudioConfig.SAMPLE_RATE, energy_threshold=0.001, min_active_frames=1)
        silent = np.zeros(AudioConfig.SAMPLE_RATE // 10, dtype=np.float32)
        voiced = np.ones(AudioConfig.SAMPLE_RATE // 10, dtype=np.float32) * 0.1

        self.assertFalse(vad.contains_voice(silent))
        self.assertTrue(vad.contains_voice(voiced))


class TestMainFunction(unittest.TestCase):
    """Test main function and CLI argument parsing."""

    def test_main_function_exists(self):
        from whimper import main

        self.assertTrue(callable(main))

    @patch('whimper.GPULiveTranscriber')
    @patch('sys.argv', ["whimper.py"])
    def test_main_imports(self, mock_transcriber_class):
        from whimper import main

        mock_transcriber = Mock()
        mock_transcriber_class.return_value.__enter__.return_value = mock_transcriber
        mock_transcriber_class.return_value.__exit__.return_value = None

        self.assertTrue(callable(main))


class IntegrationTests(unittest.TestCase):
    """Integration tests with the repository structure."""

    def test_requirements_file_exists(self):
        requirements = Path(__file__).resolve().parent.parent / 'requirements.txt'
        self.assertTrue(requirements.exists(), "requirements.txt should exist")
        content = requirements.read_text()
        self.assertIn('openai-whisper', content)
        self.assertIn('pyaudio', content)
        self.assertIn('faster-whisper', content)

    def test_src_directory_structure(self):
        src_dir = Path(__file__).resolve().parent.parent / 'src'
        self.assertTrue(src_dir.exists(), "src directory should exist")
        self.assertTrue((src_dir / 'whimper.py').exists(), "whimper.py should exist in src")

    def test_examples_directory_structure(self):
        examples_dir = Path(__file__).resolve().parent.parent / 'examples'
        self.assertTrue(examples_dir.exists(), "examples directory should exist")
        self.assertTrue((examples_dir / 'basic_usage.py').exists(), "basic_usage.py should exist")
        self.assertTrue((examples_dir / 'advanced_usage.py').exists(), "advanced_usage.py should exist")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAudioConfig))
    suite.addTest(unittest.makeSuite(TestLiveTranscriber))
    suite.addTest(unittest.makeSuite(TestTranscriptionCallback))
    suite.addTest(unittest.makeSuite(TestStreamingSession))
    suite.addTest(unittest.makeSuite(TestMainFunction))
    suite.addTest(unittest.makeSuite(IntegrationTests))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
