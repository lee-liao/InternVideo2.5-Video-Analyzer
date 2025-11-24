# InternVideo2.5 Video Analyzer

A powerful video analysis tool using the InternVideo2.5 multimodal large language model. Optimized for RTX 2070 (8GB VRAM) and 32GB RAM.

## 🚀 Features

- **Intelligent Video Analysis**: Deep understanding of video content
- **Multilingual Q&A**: Supports questions about video content
- **Memory Optimized**: Configured for consumer GPUs (8GB+ VRAM)
- **Multi-turn Conversations**: Maintains context across multiple questions
- **Real-time Progress**: Live progress indicators and performance metrics

## 📋 System Requirements

### Hardware
- **GPU**: NVIDIA RTX 2070 or better (8GB+ VRAM recommended)
- **RAM**: 16GB+ (32GB optimal)
- **Storage**: 20GB+ free space for models

### Software
- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU acceleration)

## 🛠️ Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd VLTrial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1

# Install build dependencies first
pip install packaging wheel

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

> 💡 **Follow setup steps carefully** - order matters for dependencies!

### 2. Download Model

```bash
python download_models.py
```

Choose:
- Option 1: Model code only (fast, ~50MB)
- Option 2: Complete model with weights (full functionality, ~15GB)

### 3. Verify Setup

```bash
python setup_check.py
```

### 4. Prepare Video

Place your video file as `car.mp4` in the root directory.

### 5. Run Analysis

```bash
python run_analysis.py
```

## 📁 Project Structure

```
VLTrial/
├── car.mp4                          # Your video file
├── models/                          # Model directory (downloaded)
│   └── OpenGVLab/
│       └── InternVideo2_5_Chat_8B/  # Downloaded model
├── video_analyzer.py                # Core analysis logic
├── run_analysis.py                  # Simple launcher
├── download_models.py               # Model downloader
├── setup_check.py                   # Setup verification
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```
> **Note**: The `models/` directory is ignored by Git (15GB+ of model files)

## 🎯 Usage Examples

### Quick Start
```bash
python run_analysis.py
```

### Advanced Usage
```python
from video_analyzer import VideoAnalyzer

# Initialize analyzer
analyzer = VideoAnalyzer("./models/OpenGVLab/InternVideo2_5_Chat_8B")

# Define questions
questions = [
    "Describe this video in detail.",
    "How many people appear in the video?",
    "Which part of the car is damaged?",
    "Where did the car crash?"
]

# Analyze video
results = analyzer.analyze_video("./car.mp4", questions)

# Print results
for question, answer in results.items():
    print(f"Q: {question}")
    print(f"A: {answer}")
```

## 🔧 Configuration

The script is optimized for RTX 2070 with these default settings:
- **Input Size**: 448px (balanced quality/memory)
- **Video Segments**: 64 (optimal for 30s videos)
- **Precision**: FP16 (saves VRAM)

### Customization
Modify settings in `video_analyzer.py`:
```python
class VideoAnalyzer:
    def __init__(self):
        self.input_size = 448      # Image resolution
        self.num_segments = 64     # Video segments
        self.max_new_tokens = 1024 # Response length
```

## ❓ Supported Questions

The analyzer handles various question types:

### Video Content
- "Describe this video in detail."
- "What objects are visible in the video?"
- "What actions are happening?"
- "How many people appear in the video?"

### Specific Analysis
- "Which part of the car is damaged?"
- "Where did the car crash?"
- "What color is the car?"
- "Is anyone injured in the video?"

## 🔍 Troubleshooting

### Common Issues

#### Out of Memory
- Reduce `num_segments` in `video_analyzer.py`
- Close other GPU applications
- Use smaller input size

#### Model Loading Issues
- Verify model path and files
- Run `python download_models.py` with option 3 to verify
- Ensure sufficient disk space

#### CUDA Issues
- **"No module named 'packaging'/'wheel'"**: Install build dependencies first: `pip install packaging wheel`
- **"CUDA not available / Using CPU"**: You installed CPU version of PyTorch. Reinstall: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- **"nvcc not found"**: Expected on Windows for flash-attn. It's optional and will work without it.

#### Performance Issues
- Check GPU is detected: `python setup_check.py`
- Reduce video segments for faster processing
- Monitor GPU temperature

### Platform-Specific Notes

#### Windows Users
- Flash attention requires CUDA toolkit compilation - skipped by default (OK to miss)
- Use PowerShell for activation commands
- May need to install Visual Studio Build Tools for some packages

#### macOS Users
- GPU acceleration not available (use CPU version)
- Install PyTorch with MPS support if on Apple Silicon

#### Linux Users
- Can install flash attention if CUDA toolkit is installed
- May need to install build-essential packages

### Performance Tips

1. **GPU Optimization**
   - Close unnecessary applications
   - Use high-performance GPU power settings
   - Ensure good GPU cooling

2. **Memory Management**
   - Script automatically clears GPU cache between questions
   - Monitor VRAM usage during analysis

3. **Video Settings**
   - Shorter videos process faster
   - Lower resolution videos use less memory

## 📊 Example Output

```
Starting video analysis...
InternVideo2.5 Video Analyzer
Optimized for RTX 2070 (8GB VRAM)
Using device: cuda

============================================================
VIDEO ANALYSIS STARTED
============================================================
Loading video: ./car.mp4
Video info: 916 frames, 30.00 FPS, 30.50s
Using 64 segments for analysis
✓ Video loaded and preprocessed!

============================================================
ANSWERING QUESTIONS
============================================================

Question 1: Describe this video in detail.
----------------------------------------
Answer: The video shows a white car with significant damage to its side panel...
Response time: 239.28 seconds
Memory usage: 15.62 GB
```

## 📈 Performance Benchmarks

| GPU | VRAM | Analysis Time (4 questions) |
|-----|------|----------------------------|
| RTX 2070 | 8GB | ~4 minutes |
| RTX 3080 | 10GB | ~2.5 minutes |
| RTX 4090 | 24GB | ~1.5 minutes |
| CPU | - | ~15+ minutes |

## 📝 Model Information

**Model**: InternVideo2.5_Chat_8B
- **Parameters**: 8 billion
- **Architecture**: Enhanced with long context modeling
- **Performance**:
  - MVBench: 75.7% accuracy
  - VideoMME: 65.1% accuracy
  - LongVideoBench: 60.6% accuracy

## 🔗 Dependencies

- **Core**: PyTorch, Transformers, InternVideo2.5
- **Vision**: OpenCV, Pillow, Decord
- **ML**: Einops, Timm, Accelerate
- **Utilities**: HuggingFace Hub, Sentencepiece

## 📄 License

This project uses the InternVideo2.5 model under the Apache 2.0 License.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## ⚠️ Disclaimer

- Ensure you have proper rights to analyze any video content
- Analysis results are AI-generated and should be verified
- Performance varies based on video content and hardware configuration

---

**🐛 Found an issue?** Please create an issue with your system details and error logs