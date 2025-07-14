# FTC Lens
Creating high-fidelity CAD models of FTC subsystems from images by fine-tuning VLMs. FTC Lens explores the ability of current state-of-the-art VLMs to perform visual-spatial representation tasks.

- VLM: Qwen-2.5-VL-7B (4-bit quantized)
- Training w/ Unsloth and LoRA
- Synthetic data generated with Blender, Unity, and Pyrender

The model and datasets for FTC Lens may be found on Huggingface. Running in inference requires ~12-16GB VRAM.