#!/usr/bin/env python3
"""
Video Analysis Script using InternVideo2.5
Analyze car.mp4 video and answer questions about the content.

Requirements:
- PyTorch with CUDA support
- transformers==4.40.1
- flash-attn
- decord
- opencv-python
- imageio
- torchvision

PC Configuration:
- 32GB RAM
- NVIDIA RTX 2070 (8GB VRAM)
"""

import os
import sys
import time
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Model Configuration
MODEL_PATH = "./models/OpenGVLab/InternVideo2_5_Chat_8B"
VIDEO_PATH = "./car.mp4"

class VideoAnalyzer:
    def __init__(self, model_path=None):
        """Initialize the video analyzer with InternVideo2.5 model"""
        self.model_path = model_path or MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model settings optimized for RTX 2070 (8GB VRAM)
        self.input_size = 448
        self.max_num = 1  # Reduced to save memory
        self.num_segments = 64  # Balanced for quality and performance

        # Image normalization constants
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        # Generation config
        self.generation_config = {
            'do_sample': False,
            'max_new_tokens': 1024,
            'num_beams': 1
        }

        self._load_model()
        self._build_transforms()

    def _load_model(self):
        """Load the InternVideo2.5 model"""
        print("Loading InternVideo2.5 model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Load model with memory optimization for RTX 2070
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use float16 to save memory
                low_cpu_mem_usage=True
            )

            # Move to GPU with memory optimization
            if self.device.type == 'cuda':
                self.model = self.model.to(self.device)
                # Enable gradient checkpointing to save memory
                try:
                    if hasattr(self.model, 'gradient_checkpointing_enable'):
                        self.model.gradient_checkpointing_enable()
                except Exception as e:
                    print(f"Warning: Could not enable gradient checkpointing: {e}")
                    print("Continuing without gradient checkpointing...")

            self.model.eval()
            print("✓ Model loaded successfully!")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)

    def _build_transforms(self):
        """Build image preprocessing transforms"""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((self.input_size, self.input_size),
                    interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.transform = transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio for dynamic preprocessing"""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        """Dynamically preprocess image based on aspect ratio"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        """Get frame indices for video sampling"""
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def get_num_frames_by_duration(self, duration):
        """Calculate optimal number of frames based on video duration"""
        local_num_frames = 4
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        # Optimized for RTX 2070 memory constraints
        num_frames = min(256, num_frames)  # Reduced from 512
        num_frames = max(64, num_frames)   # Reduced from 128

        return num_frames

    def load_video(self, video_path, bound=None):
        """Load and preprocess video"""
        print(f"Loading video: {video_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())
            duration = max_frame / fps

            print(f"Video info: {max_frame + 1} frames, {fps:.2f} FPS, {duration:.2f}s")

            # Calculate optimal number of segments
            num_segments = self.get_num_frames_by_duration(duration)
            print(f"Using {num_segments} segments for analysis")

            pixel_values_list = []
            num_patches_list = []

            frame_indices = self.get_index(
                bound, fps, max_frame, first_idx=0, num_segments=num_segments
            )

            for i, frame_index in enumerate(frame_indices):
                if i % 10 == 0:  # Progress indicator
                    print(f"Processing frame {i+1}/{len(frame_indices)}...")

                img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
                img = self.dynamic_preprocess(
                    img, image_size=self.input_size, use_thumbnail=True, max_num=self.max_num
                )
                pixel_values = [self.transform(tile) for tile in img]
                pixel_values = torch.stack(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)

            pixel_values = torch.cat(pixel_values_list)
            print(f"✓ Video loaded and preprocessed!")

            return pixel_values, num_patches_list

        except Exception as e:
            print(f"✗ Error loading video: {e}")
            raise

    def analyze_video(self, video_path, questions):
        """Analyze video and answer questions"""
        print(f"\n{'='*60}")
        print("VIDEO ANALYSIS STARTED")
        print(f"{'='*60}")

        # Load and preprocess video
        pixel_values, num_patches_list = self.load_video(video_path)
        pixel_values = pixel_values.to(torch.float16).to(self.device)  # Match model dtype

        print(f"Video tensor shape: {pixel_values.shape}")
        print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Create video prefix for prompts
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

        results = {}
        chat_history = None

        print(f"\n{'='*60}")
        print("ANSWERING QUESTIONS")
        print(f"{'='*60}")

        with torch.no_grad():
            for i, question in enumerate(questions, 1):
                print(f"\nQuestion {i}: {question}")
                print("-" * 40)

                # Combine video prefix with question
                full_question = video_prefix + question

                start_time = time.time()

                try:
                    # Generate response
                    output, chat_history = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        full_question,
                        self.generation_config,
                        num_patches_list=num_patches_list,
                        history=chat_history,
                        return_history=True
                    )

                    end_time = time.time()

                    print(f"Answer: {output}")
                    print(f"Response time: {end_time - start_time:.2f} seconds")
                    print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

                    results[question] = output

                except Exception as e:
                    print(f"✗ Error processing question: {e}")
                    results[question] = f"Error: {e}"

                # Clear cache between questions
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        return results

    def print_summary(self, results):
        """Print analysis summary"""
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")

        for question, answer in results.items():
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            print("-" * 40)

def main():
    """Main function"""
    print("InternVideo2.5 Video Analyzer")
    print(f"Optimized for RTX 2070 (8GB VRAM)")

    # Questions to ask about the video
    questions = [
        "Describe this video in detail.",
        "How many people appear in the video?",
        "Which part of the car is damaged?",
        "Where did the car crash?"
    ]

    try:
        # Initialize analyzer
        analyzer = VideoAnalyzer()

        # Analyze video
        results = analyzer.analyze_video(VIDEO_PATH, questions)

        # Print summary
        analyzer.print_summary(results)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()