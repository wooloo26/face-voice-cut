# face-voice-cut

Automatically batch clip video segments containing specified faces. [中文文档](docs/README_CN.md)

## Features

- **Face Feature Extraction**: Batch generate face feature files (`face_embeddings.npz`) from an image folder.
- **Video Frame Filtering**: Batch process videos to extract frames containing target faces. Supports custom frame rate, start time, output directory, and more.
- **Video Segment Clipping**: Automatically identify and clip continuous segments containing target faces. Supports frame loss tolerance, minimum segment duration, and more.

## Usage

### Generate Face Feature File

```bash
make generate
```

### Extract Video Frames

```bash
make extract
```

### Clip Video Segments

```bash
make clip
```

### Help

```bash
uv run cli --help
```

## License

MIT License
