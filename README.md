# face-voice-cut

自动批量剪辑视频中包含指定人脸的片段。

## 功能简介

- **人脸特征提取**：从图片文件夹批量生成人脸特征文件（face_embeddings.npz）。
- **视频帧筛选**：批量处理视频，提取包含目标人脸的帧，支持自定义帧率、起始时间、输出目录等参数。
- **视频片段剪辑**：自动识别并剪辑包含目标人脸的连续片段，支持容忍丢帧、最小片段时长等参数。

## 使用方法

### 1. 生成人脸特征文件

```bash
python -m fvc.cli generate -i face_original_images -o face_embeddings.npz --use-gpu
```

### 2. 提取视频帧

```bash
python -m fvc.cli extract <视频文件夹> --face-embeddings face_embeddings.npz --fps 2 -o output_face_frames --use-gpu
```

### 3. 剪辑视频片段

```bash
python -m fvc.cli clip <视频文件夹> --face-embeddings face_embeddings.npz --fps 2 -o output_video_clips --use-gpu --max-missing-frames 20 --min-clip-duration 1.0
```

## 参数说明

- `--fps`：每秒处理帧数
- `--use-gpu/--no-use-gpu`：是否启用 GPU 加速
- `--max-missing-frames`：允许连续丢失帧数
- `--min-clip-duration`：最小剪辑时长（秒）

## 示例

请将待识别的人脸图片放入 face_original_images 文件夹，视频文件放入指定文件夹，按上述命令依次运行。

## 许可证

MIT License