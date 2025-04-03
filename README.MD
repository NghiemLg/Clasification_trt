### Thứ tự chạy file có vizualization

1. main.py
2. export_resnet_to_onnx.py
3. build_trt_engine.py
4. inference_trt_engine.py (nếu dùng .engine)/ inference_resnet.py (nếu dùng .pt)



### Cấu trúc thư mục trên host:
Giả sử thư mục trên host như sau:
```
/home/user/
├── new_config.yaml   # File YAML mới để ghi đè
├── output/           # Thư mục để lưu predictions.yaml
├── engine.trt        # File engine TensorRT
├── images/           # Thư mục chứa ảnh
└── train/            # Thư mục chứa dữ liệu train
```

### File `new_config.yaml` trên host:
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

---

### Lệnh chạy với Bind Mount:
```bash
docker run --gpus all \
  -v /home/user/new_config.yaml:/app/config/config.yaml \
  -v /home/user/engine.trt:/data/engine.trt \
  -v /home/user/images:/data/images \
  -v /home/user/train:/data/train \
  -v /home/user/output:/app/output \
  tensorrt-pipeline
```

- **Giải thích**:
  - `-v /home/user/new_config.yaml:/app/config/config.yaml`: Ghi đè file YAML mặc định trong container.
  - `-v /home/user/engine.trt:/data/engine.trt`: Mount file engine từ host vào container.
  - `-v /home/user/images:/data/images`: Mount thư mục ảnh từ host.
  - `-v /home/user/train:/data/train`: Mount thư mục train từ host.
  - `-v /home/user/output:/app/output`: Mount thư mục output từ host để lưu `predictions.yaml`.

- **Kết quả**:
  - Container đọc cấu hình từ `/app/config/config.yaml` (được mount từ `/home/user/new_config.yaml`).
  - File `predictions.yaml` được ghi vào `/app/output/predictions.yaml`, và nhờ mount, nó xuất hiện trong `/home/user/output/predictions.yaml` trên host.

---

### Nếu muốn tối ưu hơn:
Nếu bạn **không muốn copy file YAML mặc định vào image** (vì luôn mount từ host), có thể bỏ dòng `COPY config/ ./config/` trong `Dockerfile`. Tuy nhiên, điều này không bắt buộc, vì mount sẽ ghi đè file mặc định anyway. Dưới đây là phiên bản tối giản:

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

- **Lưu ý**: Nếu bỏ `COPY config/`, bạn phải luôn mount file YAML khi chạy container, nếu không script sẽ báo lỗi vì không tìm thấy `/app/config/config.yaml`.
