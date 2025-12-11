#!/usr/bin/env python3
"""
ADAS Image Style Matching Test (Nano Banana)
============================================

Demonstrates ADAS prompt optimization for image generation using a
contrastive VLM judge. The policy model is Nano Banana
(`gemini-2.5-flash-image`) and the judge is a VLM (default `gpt-4o`).

Gold style references come from public HuggingFace datasets when enabled,
but by default this script uses small placeholder images so it runs fast
without any extra setup.

Usage:
    # CLI (after dataset exists)
    uvx synth-ai train --type adas --dataset products/adas/image_style_matching_dataset.json --poll

    # Or run this script directly (generates dataset + submits job)
    uv run python products/adas/test_image_style_matching.py

Requirements:
    - SYNTH_API_KEY in environment
    - GEMINI_API_KEY (or GOOGLE_API_KEY) for Nano Banana image generation
    - BACKEND_BASE_URL (optional; unset means production backend)
    - Optional for real gold images: `datasets` and `pillow`
      `pip install datasets pillow`

Env toggles:
    - USE_PLACEHOLDER_IMAGES=1 (default) uses generated placeholders.
      Set to 0 to download real gold images from HuggingFace.
"""

import os
import sys
import json
import base64
from pathlib import Path
from io import BytesIO


def load_pokemon_gold_images(num_images: int = 5) -> list[dict]:
    """Load Pokemon images from HuggingFace as gold examples.
    
    Uses a public image dataset with consistent art style.
    Falls back to creating simple placeholder images if dataset unavailable.
    """
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("Error: optional deps required for real gold images.")
        print("Install with: pip install datasets pillow")
        sys.exit(1)
    
    print(f"Loading {num_images} reference images from HuggingFace...")
    
    USE_PLACEHOLDER_IMAGES = os.environ.get("USE_PLACEHOLDER_IMAGES", "1") == "1"
    
    if USE_PLACEHOLDER_IMAGES:
        print("  Using fast placeholder images (set USE_PLACEHOLDER_IMAGES=0 for HuggingFace)")
        return _create_placeholder_images(num_images)
    
    # Try multiple public datasets (only when placeholders are disabled)
    dataset_options = [
        "nateraw/emoji",  # Small and fast
        "huggan/wikiart",
        "lambdalabs/pokemon-blip-captions",
    ]
    
    dataset = None
    dataset_name = None
    
    for ds_name in dataset_options:
        try:
            print(f"  Trying {ds_name}...")
            # Try loading with streaming first
            try:
                dataset = load_dataset(ds_name, split="train", streaming=True)
                dataset_name = ds_name
                print(f"  ✓ Successfully loaded {ds_name}")
                break
            except Exception as e:
                # If streaming fails, try non-streaming
                try:
                    dataset = load_dataset(ds_name, split="train")
                    dataset_name = ds_name
                    print(f"  ✓ Successfully loaded {ds_name} (non-streaming)")
                    break
                except Exception as e2:
                    print(f"  ✗ Failed: {e2}")
                    continue
        except Exception as e:
            print(f"  ✗ Failed to load {ds_name}: {e}")
            continue
    
    if dataset is None:
        print("  ⚠️  Could not load any dataset. Creating simple placeholder images...")
        return _create_placeholder_images(num_images)
    
    gold_images = []
    for i, example in enumerate(dataset):
        if i >= num_images:
            break
        
        # Get the image - handle different dataset formats
        if "image" in example:
            img = example["image"]
        elif "img" in example:
            img = example["img"]
        else:
            # Skip if no image field
            continue
        
        # Get caption/text if available
        caption = example.get("text", example.get("caption", example.get("label", f"Reference image {i+1}")))
        
        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img) if hasattr(img, 'shape') else Image.open(img)
        
        # Resize to reasonable size for the cookbook (256x256)
        img = img.convert("RGB")
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Encode as base64 data URL
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{img_b64}"
        
        gold_images.append({
            "image_url": data_url,
            "caption": str(caption)[:100],
        })
        print(f"  Loaded image {i+1}: {str(caption)[:50]}...")
    
    return gold_images


def _create_placeholder_images(num_images: int) -> list[dict]:
    """Create simple colored placeholder images as fallback."""
    from PIL import Image, ImageDraw
    
    gold_images = []
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]
    
    for i in range(num_images):
        # Create a simple colored square with a shape
        img = Image.new("RGB", (256, 256), colors[i % len(colors)])
        draw = ImageDraw.Draw(img)
        
        # Draw a simple shape (circle)
        center = 128
        radius = 80
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=3
        )
        
        # Encode as base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{img_b64}"
        
        gold_images.append({
            "image_url": data_url,
            "caption": f"Placeholder reference image {i+1}",
        })
        print(f"  Created placeholder image {i+1}")
    
    return gold_images


def generate_dataset(gold_images: list[dict], output_path: Path) -> dict:
    """Generate the ADAS dataset JSON with real gold images."""
    
    # Tasks: Generate Pokemon-style images of various subjects (low data: only 2 tasks)
    tasks = [
        {
            "id": "pokemon_dragon",
            "input": {
                "subject": "dragon",
                "style": "pokemon",
                "description": "A dragon creature in Pokemon art style"
            }
        },
        {
            "id": "pokemon_cat",
            "input": {
                "subject": "cat",
                "style": "pokemon", 
                "description": "A cat creature in Pokemon art style"
            }
        },
    ]
    
    # Gold outputs: Use the loaded Pokemon images as style references (low data: 3 images)
    gold_outputs = []
    
    # First, add task-specific gold outputs (link first 2 images to tasks)
    for i, (task, gold_img) in enumerate(zip(tasks, gold_images[:2])):
        gold_outputs.append({
            "task_id": task["id"],
            "output": {
                "image_url": gold_img["image_url"],
                "note": f"Reference Pokemon image - {gold_img['caption'][:100]}"
            }
        })
    
    # Then add standalone gold outputs (style references without task links)
    if len(gold_images) > 2:
        gold_outputs.append({
            "output": {
                "image_url": gold_images[2]["image_url"],
                "note": f"Standalone style reference - {gold_images[2]['caption'][:100]}"
            }
        })
    
    dataset = {
        "version": "1.0",
        "metadata": {
            "name": "nano-banana-pokemon-style-matching",
            "description": "Match Pokemon art style using Nano Banana (gemini-2.5-flash-image) with contrastive VLM judge. Gold images from lambdalabs/pokemon-blip-captions HuggingFace dataset."
        },
        "initial_prompt": "Generate an image.",
        "tasks": tasks,
        "gold_outputs": gold_outputs,
        "default_rubric": {
            "outcome": {
                "criteria": [
                    {
                        "name": "pokemon_style_match",
                        "description": "The image should match Pokemon art style: anime-inspired, colorful, cute creature design with expressive features. Score 1.0 for authentic Pokemon style, 0.5 for anime-adjacent style, 0.0 for photorealistic or non-anime style.",
                        "weight": 1.0
                    },
                    {
                        "name": "subject_recognition",
                        "description": "The creature should be clearly recognizable as the requested subject (dragon, cat, bird, etc.) while maintaining Pokemon aesthetic. Score 1.0 if subject is clear, 0.5 if somewhat recognizable, 0.0 if unclear.",
                        "weight": 0.8
                    },
                    {
                        "name": "visual_quality",
                        "description": "The image should be high quality: clean lines, vibrant colors, proper composition. Score 1.0 for high quality, 0.5 for acceptable with minor issues, 0.0 for poor quality.",
                        "weight": 0.5
                    }
                ]
            }
        },
        "judge_config": {
            "mode": "contrastive",
            "model": "gpt-4o",
            "provider": "openai"
        }
    }
    
    # Write to file
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated dataset with {len(tasks)} tasks and {len(gold_outputs)} gold outputs")
    return dataset


def main() -> None:
    """Run ADAS nano banana image style matching optimization."""
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY required")
        sys.exit(1)

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("Warning: GEMINI_API_KEY not set. Image generation may fail.")

    backend_url = os.getenv("BACKEND_BASE_URL")

    print("=" * 80)
    print("ADAS Nano Banana Pokemon Style Matching Test")
    print("=" * 80)
    print(f"Backend URL: {backend_url or '(default)'}")
    print(f"Policy Model: gemini-2.5-flash-image (Nano Banana)")
    print(f"Judge Mode: contrastive (VLM judge compares to gold images)")
    print(f"Gold Set: lambdalabs/pokemon-blip-captions (HuggingFace)")
    print()

    # Step 1: Load gold images from HuggingFace (low data: only 3 images)
    print("=" * 80)
    print("Step 1: Loading gold reference images from HuggingFace")
    print("=" * 80)
    gold_images = load_pokemon_gold_images(num_images=3)
    print(f"Loaded {len(gold_images)} gold reference images")
    print()

    # Step 2: Generate dataset JSON
    print("=" * 80)
    print("Step 2: Generating ADAS dataset")
    print("=" * 80)
    dataset_path = Path(__file__).parent / "image_style_matching_dataset.json"
    dataset_dict = generate_dataset(gold_images, dataset_path)
    print(f"Dataset saved to: {dataset_path}")
    print()

    # Step 3: Load dataset and create ADAS job
    print("=" * 80)
    print("Step 3: Creating ADAS job")
    print("=" * 80)
    
    from synth_ai.sdk import ADASJob, load_adas_taskset

    # Load dataset
    dataset = load_adas_taskset(dataset_path)
    print(f"Dataset: {dataset.metadata.name}")
    print(f"  Tasks: {len(dataset.tasks)}")
    print(f"  Gold outputs: {len(dataset.gold_outputs)}")
    print(f"  Judge mode: {dataset.judge_config.mode}")
    print(f"  Judge model: {dataset.judge_config.model}")
    print()

    # Create and submit ADAS job (low data: reduced rollout budget)
    print("Creating ADAS job...")
    job = ADASJob.from_dataset(
        dataset=dataset,
        policy_model="gemini-2.5-flash-image",  # Nano Banana
        rollout_budget=12,  # Low data: reduced from 30
        proposer_effort="medium",
        population_size=2,
        num_generations=2,  # 2 generations as requested
        backend_url=backend_url,
        api_key=api_key,
        auto_start=True,
    )

    print(f"  Policy model: {job.config.policy_model}")
    print(f"  Rollout budget: {job.config.rollout_budget}")
    print()

    print("Submitting job...")
    try:
        result = job.submit()
        print(f"  ADAS Job ID: {result.adas_job_id}")
        print(f"  Status: {result.status}")
    except RuntimeError as e:
        print(f"Error submitting job: {e}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("Streaming job progress...")
    print("=" * 80)
    print()

    try:
        final_status = job.stream_until_complete(timeout=1800.0, interval=5.0)
    except TimeoutError as e:
        print(f"Timeout: {e}")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"❌ Job ID mapping issue: {error_msg}")
            sys.exit(1)
        raise

    status = final_status.get("status") if isinstance(final_status, dict) else "unknown"
    print()
    print(f"Final status: {status}")

    if status in ("succeeded", "completed"):
        print()
        print("=" * 80)
        print("Optimized Prompt")
        print("=" * 80)
        try:
            prompt = job.download_prompt()
            print(prompt)
        except Exception as e:
            print(f"Could not download prompt: {e}")

        print()
        print("=" * 80)
        print("Test Inference")
        print("=" * 80)
        test_input = {
            "subject": "wolf",
            "style": "pokemon",
            "description": "A wolf creature in Pokemon art style"
        }
        print(f"Input: {test_input}")
        try:
            output = job.run_inference(test_input)
            print(f"Output type: {type(output)}")
            if isinstance(output, dict) and "image_url" in output:
                print(f"Generated image (data URL prefix): {output['image_url'][:80]}...")
        except Exception as e:
            print(f"Inference failed: {e}")

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
