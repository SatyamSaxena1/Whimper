"""
Test suite for Whimper live transcription functionality
"""

import unittest
import tempfile
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestAudioConfig(unittest.TestCase):
    """Test audio configuration constants"""
    
    def test_audio_config_values(self):
        """Test that audio configuration has expected values"""
        from whimper import AudioConfig
        
        self.assertEqual(AudioConfig.SAMPLE_RATE, 16000)
        self.assertEqual(AudioConfig.CHANNELS, 1)
        self.assertEqual(AudioConfig.CHUNK_SIZE, 1024)
        self.assertEqual(AudioConfig.BYTES_PER_SAMPLE, 2)

class TestLiveTranscriber(unittest.TestCase):
    """Test GPULiveTranscriber class functionality"""
    
    def setUp(self):
        \"\"\"Set up test fixtures\"\"\"
        self.mock_audio = Mock()
        self.mock_model = Mock()
    
    @patch('whimper.pyaudio.PyAudio')
    @patch('whimper.WhisperModel')
    @patch('whimper.torch.cuda.is_available')
    def test_init_success(self, mock_cuda, mock_model_class, mock_audio_class):
        \"\"\"Test successful initialization of GPULiveTranscriber\"\"\"
        # Set up mocks
        mock_cuda.return_value = False  # CPU only for testing
        mock_audio_class.return_value = self.mock_audio
        mock_model_class.return_value = self.mock_model
        
        # Mock device enumeration
        self.mock_audio.get_device_count.return_value = 2
        self.mock_audio.get_device_info_by_index.side_effect = [
            {'name': 'Device 1', 'maxInputChannels': 2},
            {'name': 'Device 2', 'maxInputChannels': 1}
        ]
        
        from whimper import GPULiveTranscriber
        
        # Create transcriber
        transcriber = GPULiveTranscriber(
            model_size=\"base\",
            language=\"en\"
        )
        
        # Verify initialization calls
        mock_model_class.assert_called_once()
        self.assertEqual(transcriber.model_size, \"base\")
        self.assertEqual(transcriber.language, \"en\")
        
        # Clean up
        transcriber.cleanup()
    
    @patch('whimper.pyaudio.PyAudio')
    @patch('whimper.WhisperModel')  
    @patch('whimper.torch.cuda.is_available')
    def test_callback_functionality(self, mock_cuda, mock_model_class, mock_audio_class):
        \"\"\"Test that callback function is called correctly\"\"\"
        # Set up mocks
        mock_cuda.return_value = False
        mock_audio_class.return_value = self.mock_audio
        mock_model_class.return_value = self.mock_model
        
        self.mock_audio.get_device_count.return_value = 1
        self.mock_audio.get_device_info_by_index.return_value = {
            'name': 'Test Device', 'maxInputChannels': 1
        }
        
        from whimper import GPULiveTranscriber
        
        # Create mock callback
        callback = Mock()
        
        # Create transcriber with callback
        transcriber = GPULiveTranscriber(callback=callback)
        
        # Verify callback is stored
        self.assertEqual(transcriber.callback, callback)
        
        # Clean up
        transcriber.cleanup()
    
    def test_audio_data_conversion(self):
        \"\"\"Test audio data conversion in callback\"\"\"
        # Create test audio data (16-bit PCM)
        test_data = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
        audio_bytes = test_data.tobytes()
        
        # Expected conversion: int16 to float32 normalized
        expected_float = test_data.astype(np.float32) / 32768.0
        
        # Test the conversion logic
        converted = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        np.testing.assert_array_almost_equal(converted, expected_float)

class TestTranscriptionCallback(unittest.TestCase):
    \"\"\"Test transcription callback function\"\"\"
    
    @patch('builtins.print')
    @patch('time.strftime')
    def test_gpu_transcription_callback(self, mock_strftime, mock_print):
        \"\"\"Test the GPU transcription callback\"\"\"
        mock_strftime.return_value = \"12:34:56\"
        
        from whimper import gpu_transcription_callback
        
        # Call the callback
        gpu_transcription_callback(\"Hello world\")
        
        # Verify it was called with correct format
        mock_print.assert_called_once_with(\"🚀🎤 [12:34:56] Hello world\")

class TestMainFunction(unittest.TestCase):
    \"\"\"Test main function and CLI argument parsing\"\"\"
    
    def test_main_function_exists(self):
        \"\"\"Test that main function exists and is callable\"\"\"
        from whimper import main
        
        self.assertTrue(callable(main))
    
    @patch('whimper.GPULiveTranscriber')
    @patch('sys.argv')
    def test_main_imports(self, mock_argv, mock_transcriber_class):
        \"\"\"Test main function imports work correctly\"\"\"
        # Mock command line arguments for basic test
        mock_argv.__getitem__.side_effect = lambda x: [
            \"whimper.py\"
        ][x] if x < 1 else []
        mock_argv.__len__.return_value = 1
        
        # Mock transcriber
        mock_transcriber = Mock()
        mock_transcriber_class.return_value.__enter__.return_value = mock_transcriber
        mock_transcriber_class.return_value.__exit__.return_value = None
        
        from whimper import main
        
        # Verify main function exists
        self.assertTrue(callable(main))

class IntegrationTests(unittest.TestCase):
    \"\"\"Integration tests with file system\"\"\"
    
    def test_requirements_file_exists(self):
        \"\"\"Test that requirements.txt exists and has expected content\"\"\"
        req_file = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        
        self.assertTrue(os.path.exists(req_file), \"requirements.txt should exist\")
        
        with open(req_file, 'r') as f:
            content = f.read()
        
        # Check for key dependencies
        self.assertIn('openai-whisper', content)
        self.assertIn('pyaudio', content)
        self.assertIn('faster-whisper', content)
    
    def test_src_directory_structure(self):
        \"\"\"Test that source directory has expected structure\"\"\"
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        whimper_file = os.path.join(src_dir, 'whimper.py')
        
        self.assertTrue(os.path.exists(src_dir), \"src directory should exist\")
        self.assertTrue(os.path.exists(whimper_file), \"whimper.py should exist in src\")
    
    def test_examples_directory_structure(self):
        \"\"\"Test that examples directory has expected files\"\"\"
        examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
        basic_example = os.path.join(examples_dir, 'basic_usage.py')
        advanced_example = os.path.join(examples_dir, 'advanced_usage.py')
        
        self.assertTrue(os.path.exists(examples_dir), \"examples directory should exist\")
        self.assertTrue(os.path.exists(basic_example), \"basic_usage.py should exist\")
        self.assertTrue(os.path.exists(advanced_example), \"advanced_usage.py should exist\")

if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAudioConfig))
    suite.addTest(unittest.makeSuite(TestLiveTranscriber))
    suite.addTest(unittest.makeSuite(TestTranscriptionCallback))
    suite.addTest(unittest.makeSuite(TestMainFunction))
    suite.addTest(unittest.makeSuite(IntegrationTests))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)