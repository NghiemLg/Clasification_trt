import os
import tensorrt as trt
import yaml
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Logger cho TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_trt_engine(config_path="config.yaml"):
    """
    Build a TensorRT engine from an ONNX model using configuration from config.yaml.
    
    Args:
        config_path (str): Path to the YAML configuration file
    """
    # Đọc config từ YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    trt_config = config.get('tensorrt', {})
    onnx_model_path = trt_config.get('onnx_model_path')
    engine_file_path = trt_config.get('engine_file_path')
    workspace_size = trt_config.get('workspace_size', 1 << 30)  # Mặc định 1GB
    enable_fp16 = trt_config.get('enable_fp16', False)
    profile = trt_config.get('optimization_profile', {})
    
    if not onnx_model_path or not engine_file_path:
        raise ValueError("Missing 'onnx_model_path' or 'engine_file_path' in tensorrt config")
    
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")
    
    # Tạo builder và network
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Parse mô hình ONNX
    logger.info(f"Parsing ONNX model from: {onnx_model_path}")
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_model_path, "rb") as model_file:
        model_data = model_file.read()
        if not parser.parse(model_data):
            logger.error("ERROR: Failed to parse ONNX model.")
            for error_idx in range(parser.num_errors):
                error = parser.get_error(error_idx)
                logger.error(f"Parser error {error_idx}: {error}")
            raise RuntimeError("Parse ONNX model failed.")
    
    # Tạo builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # Kích hoạt FP16 nếu được yêu cầu và phần cứng hỗ trợ
    if enable_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 optimization enabled.")
    
    # Thiết lập optimization profile cho input động (nếu có)
    if network.num_inputs > 0 and profile:
        input_tensor = network.get_input(0)
        if any(dim == -1 for dim in input_tensor.shape):
            profile_obj = builder.create_optimization_profile()
            min_shape = tuple(profile.get('min_shape', [1, 3, 224, 224]))
            opt_shape = tuple(profile.get('opt_shape', [4, 3, 224, 224]))
            max_shape = tuple(profile.get('max_shape', [8, 3, 224, 224]))
            profile_obj.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile_obj)
            logger.info(f"Added optimization profile for input {input_tensor.name}: "
                        f"min={min_shape}, opt={opt_shape}, max={max_shape}")
    
    # Build engine
    logger.info("Building TensorRT engine, please wait...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Build engine failed.")
    
    # Lưu engine
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    logger.info(f"TensorRT engine saved to: {engine_file_path}")

if __name__ == "__main__":
    build_trt_engine(config_path="config/config.yaml")