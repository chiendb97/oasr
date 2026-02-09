"""
OASR Python Tests

Run with: pytest tests/python/
"""

import pytest
import numpy as np


def test_import():
    """Test that the package can be imported."""
    import oasr
    assert hasattr(oasr, '__version__')
    assert hasattr(oasr, 'DataType')
    assert hasattr(oasr, 'Tensor')


@pytest.fixture
def check_cuda():
    """Check if CUDA is available."""
    import oasr
    try:
        device_count = oasr.get_device_count()
        if device_count == 0:
            pytest.skip("No CUDA devices available")
        return True
    except Exception as e:
        pytest.skip(f"CUDA not available: {e}")


class TestDataTypes:
    """Test data type enums."""
    
    def test_data_types_exist(self):
        import oasr
        assert oasr.DataType.FP32 is not None
        assert oasr.DataType.FP16 is not None
        assert oasr.DataType.BF16 is not None
        assert oasr.DataType.INT8 is not None
        assert oasr.DataType.INT32 is not None
    
    def test_memory_types_exist(self):
        import oasr
        assert oasr.MemoryType.CPU is not None
        assert oasr.MemoryType.GPU is not None
        assert oasr.MemoryType.PINNED is not None
    
    def test_attention_types_exist(self):
        import oasr
        assert oasr.AttentionType.MULTI_HEAD is not None
        assert oasr.AttentionType.RELATIVE_POSITION is not None
        assert oasr.AttentionType.ROTARY is not None
    
    def test_activation_types_exist(self):
        import oasr
        assert oasr.ActivationType.RELU is not None
        assert oasr.ActivationType.GELU is not None
        assert oasr.ActivationType.SWISH is not None


class TestTensor:
    """Test Tensor operations."""
    
    def test_create_tensor(self, check_cuda):
        import oasr
        
        t = oasr.Tensor([2, 3, 4], oasr.DataType.FP32, oasr.MemoryType.GPU, 0)
        
        assert t.shape == [2, 3, 4]
        assert t.ndim == 3
        assert t.numel == 24
        assert t.dtype == oasr.DataType.FP32
        assert t.memory_type == oasr.MemoryType.GPU
    
    def test_tensor_zeros(self, check_cuda):
        import oasr
        
        t = oasr.Tensor.zeros([4, 4], oasr.DataType.FP32, oasr.MemoryType.GPU, 0)
        
        assert t.numel == 16
        
        # Convert to numpy and check values
        arr = t.to_numpy()
        np.testing.assert_array_equal(arr, np.zeros((4, 4), dtype=np.float32))
    
    def test_tensor_ones(self, check_cuda):
        import oasr
        
        t = oasr.Tensor.ones([3, 3], oasr.DataType.FP32, oasr.MemoryType.GPU, 0)
        
        arr = t.to_numpy()
        np.testing.assert_array_equal(arr, np.ones((3, 3), dtype=np.float32))
    
    def test_tensor_from_numpy(self, check_cuda):
        import oasr
        
        np_arr = np.random.randn(4, 5).astype(np.float32)
        t = oasr.Tensor.from_numpy(np_arr, device_id=0)
        
        assert t.shape == [4, 5]
        assert t.dtype == oasr.DataType.FP32
        
        # Convert back and compare
        result = t.to_numpy()
        np.testing.assert_array_almost_equal(result, np_arr, decimal=5)
    
    def test_tensor_clone(self, check_cuda):
        import oasr
        
        t1 = oasr.Tensor.ones([2, 2], oasr.DataType.FP32, oasr.MemoryType.GPU, 0)
        t2 = t1.clone()
        
        assert t1.shape == t2.shape
        assert t1.dtype == t2.dtype
        
        # Modify t1 and ensure t2 is unaffected
        t1.fill(5.0)
        arr1 = t1.to_numpy()
        arr2 = t2.to_numpy()
        
        np.testing.assert_array_equal(arr1, np.full((2, 2), 5.0, dtype=np.float32))
        np.testing.assert_array_equal(arr2, np.ones((2, 2), dtype=np.float32))
    
    def test_tensor_reshape(self, check_cuda):
        import oasr
        
        t = oasr.Tensor([2, 3, 4], oasr.DataType.FP32, oasr.MemoryType.GPU, 0)
        t2 = t.reshape([6, 4])
        
        assert t2.shape == [6, 4]
        assert t2.numel == 24  # Same total elements
    
    def test_tensor_to_device(self, check_cuda):
        import oasr
        
        t_gpu = oasr.Tensor.ones([4, 4], oasr.DataType.FP32, oasr.MemoryType.GPU, 0)
        assert t_gpu.memory_type == oasr.MemoryType.GPU
        
        t_cpu = t_gpu.to_cpu()
        assert t_cpu.memory_type == oasr.MemoryType.CPU
        
        t_gpu2 = t_cpu.to_gpu()
        assert t_gpu2.memory_type == oasr.MemoryType.GPU
    
    def test_tensor_dtype_conversion(self, check_cuda):
        import oasr
        
        t_fp32 = oasr.Tensor.ones([4, 4], oasr.DataType.FP32, oasr.MemoryType.GPU, 0)
        assert t_fp32.dtype == oasr.DataType.FP32
        
        t_fp16 = t_fp32.to_fp16()
        assert t_fp16.dtype == oasr.DataType.FP16
        assert t_fp16.nbytes == t_fp32.numel * 2  # Half the bytes


class TestLayers:
    """Test layer configurations."""
    
    def test_attention_config(self):
        from oasr.layers import AttentionConfig
        
        config = AttentionConfig()
        config.hidden_size = 512
        config.num_heads = 8
        
        assert config.hidden_size == 512
        assert config.num_heads == 8
    
    def test_ffn_config(self):
        from oasr.layers import FFNConfig
        
        config = FFNConfig()
        config.d_model = 256
        config.d_ff = 1024
        
        assert config.d_model == 256
        assert config.d_ff == 1024
    
    def test_encoder_config(self):
        from oasr.layers import EncoderLayerConfig
        
        config = EncoderLayerConfig()
        config.d_model = 256
        config.num_heads = 4
        config.d_ff = 2048
        config.conv_kernel_size = 31
        
        assert config.d_model == 256
        assert config.conv_kernel_size == 31


class TestModels:
    """Test model configurations."""
    
    def test_model_config(self):
        from oasr.models import ModelConfig
        
        config = ModelConfig(
            encoder_type="conformer",
            num_layers=12,
            d_model=256,
            num_heads=4,
        )
        
        assert config.encoder_type == "conformer"
        assert config.num_layers == 12
        assert config.d_model == 256
    
    def test_conformer_model(self, check_cuda):
        from oasr.models import Conformer, ModelConfig
        
        config = ModelConfig(
            encoder_type="conformer",
            num_layers=2,  # Small for testing
            d_model=64,
            num_heads=2,
            d_ff=256,
        )
        
        model = Conformer(config, device_id=0)
        model.build()
        
        assert model._is_built
        assert len(model._encoder_layers) == 2


class TestUtils:
    """Test utility functions."""
    
    def test_timer(self):
        from oasr.utils import Timer
        import time
        
        with Timer("test") as t:
            time.sleep(0.01)
        
        assert t.elapsed >= 0.01
        assert t.elapsed_ms >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
