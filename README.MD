# TensorRT Pipeline README

This repository provides a pipeline for running inference with a ResNet model using TensorRT. The pipeline includes steps to export the model to ONNX, build a TensorRT engine, and perform inference with visualization. It supports both `.pt` (PyTorch) and `.engine` (TensorRT) models. This README guides you through the setup, directory structure, and execution process.

## Table of Contents

- Overview
- Directory Structure
- Configuration File
- Execution Order
- Running with Docker
- Optimizing the Dockerfile
- Notes

---

## Overview

This pipeline processes images using a ResNet model, either directly with a PyTorch `.pt` file or with a TensorRT `.engine` file for optimized inference. The pipeline includes visualization of predictions and saves results to a YAML file (`predictions.yaml`).

### Key Features

- Export a PyTorch `.pt` model to ONNX.
- Build a TensorRT engine from the ONNX model.
- Perform inference with visualization using either the `.pt` or `.engine` model.
- Configurable via a YAML file, with support for bind mounting in Docker.

---

## Directory Structure

The expected directory structure on the host machine is as follows:

```
/home/user/
├── new_config.yaml   # Configuration file to override defaults
├── output/           # Directory to store predictions.yaml
├── engine.trt        # TensorRT engine file
├── images/           # Directory containing input images
└── train/            # Directory containing training data
```

### Inside the Container

The container maps these directories using Docker bind mounts:

- `/home/user/new_config.yaml` → `/app/config/config.yaml`
- `/home/user/engine.trt` → `/data/engine.trt`
- `/home/user/images` → `/data/images`
- `/home/user/train` → `/data/train`
- `/home/user/output` → `/app/output`

---

## Configuration File

The pipeline uses a YAML configuration file (`new_config.yaml`) to specify paths and settings. Below is an example of `new_config.yaml`:

```yaml
tensorrt:
  engine_file_path: "/data/engine.trt"
inference:
  image_dir: "/data/images"
  image_size: [224, 224]
  output_yaml: "/app/output/predictions.yaml"
preprocess_data:
  train_dir: "/data/train"
```

### Explanation of Fields

- `tensorrt.engine_file_path`: Path to the TensorRT engine file inside the container.
- `inference.image_dir`: Directory containing input images for inference.
- `inference.image_size`: Size of images after preprocessing (e.g., `[224, 224]` for ResNet).
- `inference.output_yaml`: Path where the inference results (`predictions.yaml`) will be saved.
- `preprocess_data.train_dir`: Directory containing training data (used for preprocessing if needed).

---

## Execution Order

The pipeline consists of several scripts that must be run in the following order:

1. `main.py`: The entry point to execute the entire pipeline. It orchestrates the other scripts.
2. `export_resnet_to_onnx.py`: Converts the PyTorch `.pt` model to ONNX format.
3. `build_trt_engine.py`: Builds a TensorRT engine from the ONNX model.
4. **Inference**:
   - If using a TensorRT engine (`.engine`): Run `inference_trt_engine.py`.
   - If using the PyTorch model (`.pt`): Run `inference_resnet.py`.

### Note

- `inference_trt_engine.py` and `inference_resnet.py` include visualization of predictions (e.g., drawing bounding boxes or class labels on images).
- Results are saved to `/app/output/predictions.yaml` (mapped to `/home/user/output/predictions.yaml` on the host).

---

## Running with Docker

The pipeline is containerized using Docker, leveraging NVIDIA TensorRT for GPU acceleration. Use the following command to run the pipeline with bind mounts:

```bash
docker run --gpus all \
  -v /home/user/new_config.yaml:/app/config/config.yaml \
  -v /home/user/engine.trt:/data/engine.trt \
  -v /home/user/images:/data/images \
  -v /home/user/train:/data/train \
  -v /home/user/output:/app/output \
  tensorrt-pipeline
```

### Explanation of Bind Mounts

- `-v /home/user/new_config.yaml:/app/config/config.yaml`: Overrides the default configuration file in the container with the host's `new_config.yaml`.
- `-v /home/user/engine.trt:/data/engine.trt`: Mounts the TensorRT engine file from the host to the container.
- `-v /home/user/images:/data/images`: Mounts the directory containing input images.
- `-v /home/user/train:/data/train`: Mounts the directory containing training data.
- `-v /home/user/output:/app/output`: Mounts the output directory to save `predictions.yaml`.

### Expected Output

- The container reads the configuration from `/app/config/config.yaml`.
- Inference results are saved to `/app/output/predictions.yaml`, which appears on the host at `/home/user/output/predictions.yaml`.

---

## Optimizing the Dockerfile

The default `Dockerfile` copies a `config/` directory into the container. However, since the pipeline always mounts `new_config.yaml` from the host, you can optimize the `Dockerfile` by removing the `COPY config/` step. Below is the minimal `Dockerfile`:

```dockerfile
FROM nvcr.io/nvidia/tensorrt:25.03-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python3", "src/inference.py"]
```

### Why Optimize?

- **Avoid Redundancy**: If you always mount `new_config.yaml` from the host, there's no need to copy a default `config/` directory into the image.
- **Smaller Image Size**: Removing the `COPY config/` step reduces the image size slightly.

### Caveat

- If you remove `COPY config/`, you **must** mount a configuration file when running the container (e.g., `-v /home/user/new_config.yaml:/app/config/config.yaml`). Otherwise, the script will fail because `/app/config/config.yaml` will not exist.

---

## Notes

- **GPU Support**: Ensure your host machine has an NVIDIA GPU and the NVIDIA Container Toolkit installed to use the `--gpus all` flag.
- **File Paths**: Double-check the paths in `new_config.yaml` to match the container's filesystem (e.g., `/data/engine.trt` instead of `/home/user/engine.trt`).
- **Visualization**: The inference scripts (`inference_trt_engine.py` and `inference_resnet.py`) include visualization. Ensure you have a display server (e.g., X11) if running visualization inside the container, or modify the scripts to save visualized images instead.
- **Fallback Option**: If you don't have a TensorRT engine, you can use `inference_resnet.py` to run inference directly with the `.pt` model, though this will be slower than using the `.engine` file.

---

## Troubleshooting

- **Error: "Cannot find /app/config/config.yaml"**:

  - Ensure you mounted the configuration file using `-v /home/user/new_config.yaml:/app/config/config.yaml`.
  - If you modified the `Dockerfile` to remove `COPY config/`, this file must be mounted.

- **Error: "No images found in /data/images"**:

  - Verify that the `/home/user/images` directory on the host contains images and is correctly mounted.

- **Error: "Failed to load engine.trt"**:

  - Check that `/home/user/engine.trt` exists on the host and is mounted correctly.

For further assistance, please open an issue in the repository.