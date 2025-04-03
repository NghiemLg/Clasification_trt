import os
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import yaml
import logging
from PIL import Image
from torchvision import transforms

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Logger cho TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class TensorRTInference:
    def __init__(self, engine_path: str):
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        logger.info(f"Initialized TensorRT engine with {len(self.inputs)} input(s), {len(self.outputs)} output(s).")

    def _load_engine(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found at {engine_path}")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        try:
            engine = self.runtime.deserialize_cuda_engine(engine_data)
            logger.info(f"Loaded TensorRT engine from {engine_path}")
            return engine
        except Exception as e:
            logger.error(f"Error deserializing engine: {e}")
            return None

    class HostDeviceMem:
        def __init__(self, name: str, host_mem, device_mem, shape):
            self.name = name
            self.host = host_mem
            self.device = device_mem
            self.shape = shape

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        profile_index = 0

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            tensor_shape = list(self.engine.get_tensor_shape(tensor_name))
            if any(dim == -1 for dim in tensor_shape):
                try:
                    min_shape, opt_shape, max_shape = self.engine.get_profile_shape(profile_index, tensor_name)
                    tensor_shape = list(max_shape)
                    logger.info(f"Tensor '{tensor_name}' has dynamic shape, using max_shape {tensor_shape}")
                except AttributeError:
                    logger.warning(f"Cannot get profile shape for '{tensor_name}'. Replacing -1 with 1.")
                    tensor_shape = [(1 if dim == -1 else dim) for dim in tensor_shape]

            vol = trt.volume(tensor_shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            host_mem = cuda.pagelocked_empty(vol, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if tensor_mode == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(tensor_name, host_mem, device_mem, tensor_shape))
            else:
                outputs.append(self.HostDeviceMem(tensor_name, host_mem, device_mem, tensor_shape))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        if not isinstance(input_data, (list, tuple)):
            input_data_list = [input_data]
        else:
            input_data_list = list(input_data)

        for inp, data in zip(self.inputs, input_data_list):
            arr = np.ascontiguousarray(data)
            if arr.nbytes > inp.host.nbytes:
                logger.warning(f"Input '{inp.name}' size exceeds allocated buffer. Reallocating.")
                inp.host = cuda.pagelocked_empty(arr.size, arr.dtype)
                inp.device = cuda.mem_alloc(arr.nbytes)
                bind_index = self.engine.get_binding_index(inp.name)
                self.bindings[bind_index] = int(inp.device)
            np.copyto(inp.host, arr.ravel())
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            if any(dim == -1 for dim in self.engine.get_tensor_shape(inp.name)):
                new_shape = tuple(arr.shape)
                self.context.set_input_shape(inp.name, new_shape)
                logger.info(f"Set shape for input '{inp.name}' = {new_shape}")

        missing = self.context.infer_shapes()
        if missing:
            raise RuntimeError(f"Shapes not specified for tensors: {missing}")

        for i, ptr in enumerate(self.bindings):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, ptr)

        exec_success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not exec_success:
            raise RuntimeError("Inference failed during execute_async_v3.")

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()

        results = []
        for out in self.outputs:
            actual_shape = tuple(self.context.get_tensor_shape(out.name))
            if any(dim == -1 for dim in out.shape) or actual_shape != tuple(out.shape):
                out.shape = actual_shape
                logger.info(f"Updated actual shape for output '{out.name}' = {actual_shape}")
            result_array = np.array(out.host).reshape(out.shape)
            results.append(result_array)
        return results

def load_image(image_path, image_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    pil_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(pil_image).unsqueeze(0)
    return image_tensor.numpy()

def main(config_path="/app/config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trt_config = config.get('tensorrt', {})
    inference_config = config.get('inference', {})
    preprocess_config = config.get('preprocess_data', {})

    # Lấy tham số từ YAML
    engine_path = trt_config.get('engine_file_path')
    image_dir = inference_config.get('image_dir')
    image_size = tuple(inference_config.get('image_size', [224, 224]))
    train_dir = preprocess_config.get('train_dir')
    output_yaml = inference_config.get('output_yaml', '/app/output/predictions.yaml')

    if not engine_path or not image_dir or not train_dir:
        raise ValueError("Missing required parameters in YAML: 'engine_file_path', 'image_dir', or 'train_dir'")

    # Tạo thư mục chứa file YAML nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)

    # Inference với .engine
    class_names = sorted(os.listdir(train_dir))
    trt_infer = TensorRTInference(engine_path)

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    logger.info(f"Found {len(image_paths)} images for inference")
    predictions = []

    for img_path in image_paths:
        input_data = load_image(img_path, image_size)
        logger.info(f"Running inference on {os.path.basename(img_path)}...")
        result = trt_infer.infer(input_data)

        logits = result[0]
        predicted_idx = np.argmax(logits, axis=1)[0]
        predicted_class = class_names[predicted_idx]

        logger.info(f"Predicted class for {os.path.basename(img_path)}: {predicted_class}")
        predictions.append({
            "filename": os.path.basename(img_path),
            "predicted_class": predicted_class
        })

    # Lưu kết quả vào file YAML
    with open(output_yaml, 'w') as f:
        yaml.safe_dump(predictions, f)
    logger.info(f"Saved predictions to {output_yaml}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program failed: {e}")
        raise