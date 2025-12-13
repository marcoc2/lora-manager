import sys
import shutil
import subprocess
from pathlib import Path
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMessageBox

# Import utility for saving config (assuming it's in a shared module or we move it here)
# For now, we'll implement save_config logic within the controller or keep using the existing one if it's external.
# Based on previous file reads, save_config was imported from *_widgets_base.py.
# We should probably move that utility or reimplement it.
# Let's assume we can import it or just use json/toml directly.
import json

class TrainingController(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.script_manager = None # Initialize if needed, or pass in

    def set_script_manager(self, script_manager):
        self.script_manager = script_manager

    # --- Flux Logic (ai-toolkit) ---

    def get_flux_command(self, config, dataset_path):
        """Generates the command for Flux training using ai-toolkit YAML config"""
        import yaml
        import re

        # Validate inputs
        if not config.get("model_name_or_path"):
            return None, "Model path is required"

        if not config.get("output_dir"):
            return None, "Output directory is required"

        if not config.get("output_name"):
            return None, "Output name is required"

        # Validate resume checkpoint if specified
        if config.get("resume_training"):
            resume_path = config.get("resume_path")
            if not resume_path:
                return None, "Resume training is enabled but no checkpoint selected"
            if not Path(resume_path).exists():
                return None, f"Resume checkpoint not found: {resume_path}"

        # Update config with dataset_path and save it
        try:
            config["dataset_path"] = str(dataset_path).replace("\\", "/")
            with open("flux_config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to update config with dataset path: {e}")

        # Prepare output directory
        output_dir = Path(config.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure name is stripped of extension
        name_no_ext = Path(config.get("output_name", "flux_lora")).stem

        # Build network config with resume support
        network_config = {
            "type": "lora",
            "linear": config.get("network_dim", 16),
            "linear_alpha": config.get("network_alpha", 16)
        }

        # Handle resume training
        if config.get("resume_training") and config.get("resume_path"):
            resume_path = Path(config.get("resume_path"))
            network_config["resume_lora_path"] = str(resume_path.absolute()).replace("\\", "/")

            # Copy checkpoint to expected location for ai-toolkit auto-resume
            job_name = name_no_ext
            match = re.search(r'_(\d+)\.(safetensors|pt)$', resume_path.name)
            if match:
                step_num = match.group(1)
                expected_name = f"{job_name}_{step_num}.safetensors"
                expected_path = output_dir / expected_name

                if expected_path != resume_path.absolute():
                    shutil.copy2(resume_path, expected_path)
                    print(f"Checkpoint copied to: {expected_path}")

        # Build EMA config
        ema_config = {
            "use_ema": config.get("use_ema", True),
            "ema_decay": config.get("ema_decay", 0.99)
        }

        # Get resolution list
        resolution = config.get("resolution", [512, 768, 1024])
        if not isinstance(resolution, list):
            resolution = [resolution]

        # Parse sample prompts
        prompts = [p.strip() for p in config.get("sample_prompts", "").split("\n") if p.strip()]
        if not prompts:
            prompts = ["a person in the park, high quality photo"]

        # Generate YAML config
        yaml_config = {
            "job": "extension",
            "config": {
                "name": name_no_ext,
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": str(output_dir.absolute()).replace("\\", "/"),
                    "device": "cuda:0",
                    "network": network_config,
                    "save": {
                        "dtype": config.get("save_precision", "float16"),
                        "save_every": config.get("save_every", 250),
                        "max_step_saves_to_keep": 4,
                        "push_to_hub": False
                    },
                    "datasets": [{
                        "folder_path": str(dataset_path.absolute()).replace("\\", "/"),
                        "caption_ext": "txt",
                        "caption_dropout_rate": config.get("caption_dropout_rate", 0.05),
                        "shuffle_tokens": False,
                        "cache_latents_to_disk": config.get("cache_latents_to_disk", True),
                        "resolution": resolution
                    }],
                    "train": {
                        "batch_size": config.get("batch_size", 1),
                        "steps": config.get("steps", 2000),
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": config.get("optimizer", "adamw8bit"),
                        "lr": float(config.get("learning_rate", "1e-4")),
                        "ema_config": ema_config,
                        "dtype": config.get("mixed_precision", "bf16"),
                        "disable_sampling": not config.get("enable_sampling", True)
                    },
                    "model": {
                        "name_or_path": config.get("model_name_or_path"),
                        "is_flux": True,
                        "quantize": config.get("quantize", True),
                        "low_vram": config.get("low_vram", False)
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": config.get("sample_every", 250),
                        "width": config.get("sample_width", 1024),
                        "height": config.get("sample_height", 1024),
                        "prompts": prompts,
                        "neg": "",
                        "seed": config.get("seed", 42),
                        "walk_seed": True,
                        "guidance_scale": config.get("guidance_scale", 4),
                        "sample_steps": config.get("sample_steps", 20)
                    }
                }],
                "meta": {
                    "name": "[name]",
                    "version": "1.0"
                }
            }
        }

        # Add trigger word if specified
        if config.get("trigger_word"):
            yaml_config["config"]["process"][0]["trigger_word"] = config.get("trigger_word")

        # Save YAML file
        yaml_path = output_dir / "flux_train.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f, sort_keys=False)

        # Build command
        toolkit_path = Path("reference/ai-toolkit-original").absolute()
        run_script = toolkit_path / "run.py"

        # Use the specific venv python
        venv_python = Path("C:/Apps/sd-scripts/venv/Scripts/python.exe")
        if venv_python.exists():
            python_exe = str(venv_python)
        else:
            python_exe = sys.executable

        command = f'"{python_exe}" "{run_script}" "{yaml_path}"'

        return command, None

    # --- Qwen Logic ---

    def validate_qwen_inputs(self, config):
        errors = []
        
        if not config.get("python_venv_path"):
            errors.append("Python/Venv path is required")
        elif not Path(config["python_venv_path"]).exists():
            errors.append("Python executable does not exist at specified path")
        
        if not config.get("dit_path"):
            errors.append("DiT model path is required")
        elif not Path(config["dit_path"]).exists():
            errors.append("DiT model file does not exist")
            
        if not config.get("text_encoder_path"):
            errors.append("Text encoder path is required")
        elif not Path(config["text_encoder_path"]).exists():
            errors.append("Text encoder file does not exist")
            
        if not config.get("vae_path"):
            errors.append("VAE model path is required")
        elif not Path(config["vae_path"]).exists():
            errors.append("VAE model file does not exist")
            
        if not config.get("musubi_dir"):
            errors.append("Musubi Tuner directory is required")
        elif not Path(config["musubi_dir"]).exists():
            errors.append("Musubi Tuner directory does not exist")
            
        return errors

    def prepare_musubi_dataset(self, dataset_path, musubi_dir):
        """Prepares the dataset structure required by Musubi"""
        musubi_dir = Path(musubi_dir)
        musubi_cache_dir = musubi_dir / "cache_imgs"
        musubi_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine source cropped images directory
        if (dataset_path / "dataset.toml").exists():
             # dataset_path is likely the artifact folder itself
             cropped_dir = dataset_path
             captions_dir = dataset_path / "captions" # Assuming captions are here? Or in cropped_images/captions? 
             # Let's check typical structure: root/cropped_images/captions
             # If dataset_path is root/cropped_images, then captions is root/cropped_images/captions
        else:
             cropped_dir = dataset_path / "cropped_images"
             captions_dir = cropped_dir / "captions"

        if not cropped_dir.exists():
            return # Should handle error

        # Copy images
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        for img_file in cropped_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                dest_img = musubi_dir / img_file.name
                if not dest_img.exists():
                    shutil.copy2(img_file, dest_img)
        
        # Copy captions with encoding fix
        if captions_dir.exists():
            for caption_file in captions_dir.glob("*.txt"):
                dest_caption = musubi_dir / caption_file.name
                if not dest_caption.exists():
                    try:
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(caption_file, 'r', encoding='latin-1') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            with open(caption_file, 'r', encoding='cp1252') as f:
                                content = f.read()
                    
                    with open(dest_caption, 'w', encoding='utf-8') as f:
                        f.write(content.rstrip() + '\n')

    def generate_musubi_toml(self, config, musubi_dir):
        import toml
        
        musubi_toml = {
            "general": {
                "resolution": 512, # Should this be configurable?
                "caption_extension": ".txt",
                "batch_size": config.get('batch_size', 1),
                "enable_bucket": True,
                "bucket_no_upscale": False
            },
            "datasets": [{
                "image_directory": str(musubi_dir.resolve()),
                "cache_directory": str((musubi_dir / "cache_imgs").resolve()),
                "num_repeats": 1 # Should be configurable?
            }]
        }
        
        toml_path = musubi_dir / "dataset_qwen.toml"
        with open(toml_path, "w", encoding='utf-8') as f:
            toml.dump(musubi_toml, f)
        
        return toml_path

    def get_qwen_command(self, config, dataset_path):
        errors = self.validate_qwen_inputs(config)
        if errors:
            return None, errors

        # Prepare Musubi structure
        # We'll create a parallel folder for musubi specific structure if needed
        # Or use a temp folder? The original code used 'cropped_images_musubi' inside dataset_path
        if (dataset_path / "dataset.toml").exists():
             # dataset_path is the artifact folder
             parent_dir = dataset_path.parent
             musubi_dir = parent_dir / f"{dataset_path.name}_musubi"
        else:
             musubi_dir = dataset_path / "cropped_images_musubi"
        
        musubi_dir.mkdir(parents=True, exist_ok=True)
        self.prepare_musubi_dataset(dataset_path, musubi_dir)
        
        dataset_config = musubi_dir / "dataset_qwen.toml"
        if not dataset_config.exists():
            self.generate_musubi_toml(config, musubi_dir)

        # Check cache requirements
        cache_dir = musubi_dir / "cache_imgs"
        needs_latent_cache = not any(cache_dir.glob("*.npz"))
        needs_text_cache = not any(cache_dir.glob("*.txt"))

        cache_commands = []
        if needs_latent_cache or needs_text_cache:
            # We return a special structure to indicate cache is needed
            # The View/MainController should handle the user prompt
            
            if needs_latent_cache:
                latent_cache_script = Path(config["musubi_dir"]) / "src" / "musubi_tuner" / "qwen_image_cache_latents.py"
                cache_commands.append([
                    config["python_venv_path"],
                    str(latent_cache_script),
                    "--vae", config["vae_path"],
                    "--dataset_config", str(dataset_config)
                ])
            
            if needs_text_cache:
                text_cache_script = Path(config["musubi_dir"]) / "src" / "musubi_tuner" / "qwen_image_cache_text_encoder_outputs.py"
                text_cmd = [
                    config["python_venv_path"],
                    str(text_cache_script),
                    "--text_encoder", config["text_encoder_path"],
                    "--dataset_config", str(dataset_config)
                ]
                if config["fp8_llm"]:
                    text_cmd.extend(["--device", "cpu"])
                cache_commands.append(text_cmd)

        # Build training command
        python_path = Path(config["python_venv_path"])
        venv_dir = python_path.parent
        accelerate_exe = venv_dir / "accelerate.exe"
        
        if accelerate_exe.exists():
            command_start = [str(accelerate_exe), "launch"]
        else:
            command_start = [str(python_path), "-m", "accelerate", "launch"]
            
        musubi_script = Path(config["musubi_dir"]) / "src" / "musubi_tuner" / "qwen_image_train_network.py"
        
        cmd = command_start + [
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", config["mixed_precision"],
            str(musubi_script),
            "--dit", config["dit_path"],
            "--vae", config["vae_path"],
            "--text_encoder", config["text_encoder_path"],
            "--dataset_config", str(dataset_config),
            "--output_dir", config["output_dir"],
            "--output_name", config["output_name"],
            "--network_module", "networks.lora_qwen_image",
            "--fp8_base",
            "--network_dim", str(config["network_dim"]),
            "--learning_rate", config["learning_rate"],
            "--timestep_sampling", config["timestep_sampling"],
            "--weighting_scheme", config["weighting_scheme"],
            "--discrete_flow_shift", str(config["discrete_flow_shift"]),
            "--optimizer_type", config["optimizer_type"],
            "--max_train_epochs", str(config["epochs"]),
            "--save_every_n_epochs", str(config["save_every"]),
            "--seed", "42"
        ]

        if config["gradient_checkpointing"]: cmd.append("--gradient_checkpointing")
        if config["sdpa"]: cmd.append("--sdpa")
        if config["fp8_llm"]: cmd.append("--fp8_llm")
        if config["blocks_to_swap"] > 0: cmd.extend(["--blocks_to_swap", str(config["blocks_to_swap"])])
        if config["flip_aug"]: cmd.append("--flip_aug")
        if config["resume_training"] and config["resume_path"]:
            cmd.extend(["--network_weights", config["resume_path"]])

        return {
            "training_command": cmd,
            "cache_commands": cache_commands,
            "needs_cache": bool(cache_commands)
        }, None

    def get_cache_latents_command(self, config, dataset_path):
        # Similar logic to prepare dataset and get command
        # Simplified for brevity, reuse logic from get_qwen_command
        pass # To be implemented if needed separately

    # --- Z-Image Logic ---

    def get_zimage_command(self, config, dataset_path):
        """Generates the command for Z-Image training using ai-toolkit"""
        import yaml
        
        # Validate inputs
        if not config.get("model_path"):
            return None, "Model path is required"

        # Validate resume checkpoint if specified
        if config.get("resume_training"):
            resume_path = config.get("resume_path")
            if not resume_path:
                return None, "Resume training is enabled but no checkpoint selected"
            if not Path(resume_path).exists():
                return None, f"Resume checkpoint not found: {resume_path}"

        # Update config with dataset_path and save it (as requested by user)
        try:
            config["dataset_path"] = str(dataset_path).replace("\\", "/")
            with open("zimage_training_config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to update config with dataset path: {e}")
        
        # Prepare output directory
        output_dir = Path(config.get("output_dir", "output"))
        # ai-toolkit creates a folder with "name" inside "training_folder", so we don't append name here
        # if config.get("output_name"):
        #     output_dir = output_dir / config.get("output_name")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure name is stripped of extension
        name_no_ext = Path(config.get("output_name", "zimage_lora")).stem

        # Build network config with resume support
        network_config = {
            "type": "lora",
            "linear": config.get("rank", 16),
            "linear_alpha": config.get("alpha", 16)
        }

        # Handle resume training (Dual Approach)
        if config.get("resume_training") and config.get("resume_path"):
            resume_path = Path(config.get("resume_path"))

            # Approach 1: Add to YAML (may not work in all ai-toolkit versions)
            network_config["resume_lora_path"] = str(resume_path.absolute()).replace("\\", "/")

            # Approach 2: Copy checkpoint to expected location (guaranteed to work)
            # ai-toolkit auto-resume will find the checkpoint
            # Since we modified BaseTrainProcess to save directly to training_folder (no nested folder),
            # we copy checkpoint directly to output_dir
            job_name = name_no_ext

            # Extract step from checkpoint source
            import re
            match = re.search(r'_(\d+)\.(safetensors|pt)$', resume_path.name)
            if match:
                step_num = match.group(1)
                expected_name = f"{job_name}_{step_num}.safetensors"
                expected_path = output_dir / expected_name

                # Copy if not already in expected location
                if expected_path != resume_path.absolute():
                    import shutil
                    shutil.copy2(resume_path, expected_path)
                    print(f"Checkpoint copied to: {expected_path}")
                    print(f"ai-toolkit will resume from step {step_num}")

        # Generate YAML config
        yaml_config = {
            "job": "extension",
            "config": {
                "name": name_no_ext,
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": str(output_dir.absolute()),
                    "device": "cuda:0",
                    "network": network_config,
                    "save": {
                        "dtype": "float16",
                        "save_every": config.get("save_every", 250),
                        "max_step_saves_to_keep": 4,
                        "push_to_hub": False
                    },
                    "datasets": [{
                        "folder_path": str(dataset_path.absolute()),
                        "caption_ext": "txt",
                        "caption_dropout_rate": 0.05,
                        "shuffle_tokens": False,
                        "cache_latents_to_disk": True,
                        "cache_text_embeddings": True, # Added for VRAM optimization
                        "resolution": [config.get("resolution", 1024)]
                    }],
                    "train": {
                        "batch_size": config.get("batch_size", 1),
                        "steps": config.get("steps", 2000),
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": float(config.get("learning_rate", "1e-4")),
                        "ema_config": {
                            "use_ema": False, # Disabled for VRAM optimization
                            "ema_decay": 0.99
                        },
                        "dtype": "bf16",
                        # Enable/disable sampling based on UI checkbox (inverted logic)
                        "disable_sampling": not config.get("enable_sampling", True)
                    },
                    "model": {
                        "arch": "zimage",
                        "name_or_path": config.get("model_path"),
                        "extras_name_or_path": config.get("text_encoder_path"),
                        "tokenizer_path": config.get("tokenizer_path"),
                        "is_flux": False, # Z-Image is not Flux
                        "quantize": True,
                        "low_vram": True, # Added for VRAM optimization
                        "assistant_lora_path": config.get("adapter_path")
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": config.get("sample_every", 250),
                        "width": 1024,
                        "height": 1024,
                        "prompts": [p for p in config.get("sample_prompts", "").split("\n") if p.strip()],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": False,
                        "guidance_scale": 1.0,
                        "sample_steps": 8
                    }
                }],
                "meta": {
                    "name": "[name]",
                    "version": "1.0"
                }
            }
        }
        
        # Save YAML file
        yaml_path = output_dir / "zimage_train.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f, sort_keys=False)
            
        # Build command
        # Assuming ai-toolkit is in reference/ai-toolkit
        # We need to run run.py from that directory
        toolkit_path = Path("reference/ai-toolkit-original").absolute()
        
        # We need to run this in a python environment that has ai-toolkit dependencies
        # Using sys.executable ensures we use the same python that is running this app
        # which is likely the one where the user installed the requirements.
        
        run_script = toolkit_path / "run.py"
        
        # Use the specific venv python
        venv_python = Path("C:/Apps/sd-scripts/venv/Scripts/python.exe")
        if venv_python.exists():
            python_exe = str(venv_python)
        else:
            python_exe = sys.executable
        
        command = f'"{python_exe}" "{run_script}" "{yaml_path}"'

        return command, None

    # --- Wan Logic ---

    def get_wan_command(self, config, dataset_path):
        """Generates the command for Wan 2.1/2.2 training using ai-toolkit YAML config"""
        import yaml

        # Validate inputs
        if not config.get("output_dir"):
            return None, "Output directory is required"

        if not config.get("output_name"):
            return None, "Output name is required"

        # Prepare output directory
        output_dir = Path(config.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get model version info
        version = config.get("model_version", "wan21_14b")
        is_wan22 = version == "wan22_14b"
        is_wan21_1b = version == "wan21_1b"

        # Ensure name is stripped of extension
        name_no_ext = Path(config.get("output_name", "wan_lora")).stem

        # Determine resolution format
        # For video/Wan: [height, width] - Wan uses height first
        if config.get("training_mode") == "video":
            resolution = [config.get("resolution_height", 480), config.get("resolution_width", 832)]
        else:
            resolution = [config.get("resolution_width", 832)]

        # Build network config
        network_config = {
            "type": "lora",
            "linear": config.get("lora_rank", 32),
            "linear_alpha": config.get("lora_alpha", 32)
        }

        # Build datasets config
        datasets_config = [{
            "folder_path": str(dataset_path.absolute()).replace("\\", "/"),
            "caption_ext": "txt",
            "caption_dropout_rate": 0.05,
            "shuffle_tokens": False,
            "cache_latents_to_disk": True,
            "resolution": resolution
        }]

        # Add num_frames for video or Wan 2.2 image training
        if config.get("training_mode") == "video":
            datasets_config[0]["num_frames"] = config.get("num_frames", 40)
        elif is_wan22:
            # Wan 2.2 requires num_frames even for image training (set to 1)
            datasets_config[0]["num_frames"] = 1

        # Build train config
        train_config = {
            "batch_size": config.get("batch_size", 1),
            "steps": config.get("steps", 2000),
            "gradient_accumulation": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": config.get("gradient_checkpointing", True),
            "noise_scheduler": "flowmatch",
            "optimizer": "adamw8bit",
            "lr": float(config.get("learning_rate", "1e-4")),
            "optimizer_params": {
                "weight_decay": 1e-4
            },
            "dtype": "bf16"
        }

        # Version-specific train settings
        if is_wan22:
            train_config["timestep_type"] = "linear"
            train_config["switch_boundary_every"] = config.get("switch_boundary_every", 10)
            if config.get("cache_text_embeddings", True):
                train_config["cache_text_embeddings"] = True
        else:
            train_config["timestep_type"] = "sigmoid"
            if config.get("unload_text_encoder", True):
                train_config["unload_text_encoder"] = True

        # EMA config
        if config.get("use_ema", True):
            train_config["ema_config"] = {
                "use_ema": True,
                "ema_decay": config.get("ema_decay", 0.99)
            }

        # Disable sampling if not enabled
        if not config.get("enable_sampling", True):
            train_config["disable_sampling"] = True

        # Build model config
        model_config = {
            "name_or_path": config.get("name_or_path"),
            "arch": config.get("arch", "wan21")
        }

        # Memory optimization settings
        if config.get("quantize", True):
            model_config["quantize"] = True
            if is_wan22:
                # Wan 2.2 uses special quantization with accuracy recovery adapter
                model_config["qtype"] = "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors"

        if config.get("quantize_te", True):
            model_config["quantize_te"] = True
            if is_wan22:
                model_config["qtype_te"] = "qfloat8"

        if config.get("low_vram", True) and not is_wan21_1b:
            model_config["low_vram"] = True

        # Wan 2.2 specific model_kwargs
        if is_wan22:
            model_config["model_kwargs"] = {
                "train_high_noise": config.get("train_high_noise", True),
                "train_low_noise": config.get("train_low_noise", True)
            }

        # Build sample config
        sample_config = {
            "sampler": "flowmatch",
            "sample_every": config.get("sample_every", 250),
            "width": config.get("resolution_width", 832),
            "height": config.get("resolution_height", 480),
            "num_frames": config.get("num_frames", 1) if config.get("training_mode") == "video" else 1,
            "fps": config.get("fps", 16),
            "neg": "",
            "seed": 42,
            "walk_seed": True,
            "guidance_scale": config.get("guidance_scale", 5.0),
            "sample_steps": config.get("sample_steps", 30)
        }

        # Parse sample prompts
        prompts = [p.strip() for p in config.get("sample_prompts", "").split("\n") if p.strip()]
        if not prompts:
            prompts = ["a person walking in the park"]  # Default prompt
        sample_config["prompts"] = prompts

        # Build process config
        process_config = {
            "type": "sd_trainer",
            "training_folder": str(output_dir.absolute()).replace("\\", "/"),
            "device": "cuda:0",
            "network": network_config,
            "save": {
                "dtype": "float16",
                "save_every": config.get("save_every", 250),
                "max_step_saves_to_keep": 4,
                "push_to_hub": False
            },
            "datasets": datasets_config,
            "train": train_config,
            "model": model_config,
            "sample": sample_config
        }

        # Add trigger word if specified
        if config.get("trigger_word"):
            process_config["trigger_word"] = config.get("trigger_word")

        # Build full YAML config
        yaml_config = {
            "job": "extension",
            "config": {
                "name": name_no_ext,
                "process": [process_config],
                "meta": {
                    "name": "[name]",
                    "version": "1.0"
                }
            }
        }

        # Save YAML file
        yaml_path = output_dir / f"{name_no_ext}_train.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f, sort_keys=False, default_flow_style=False)

        # Build command
        toolkit_path = Path("reference/ai-toolkit-original").absolute()
        run_script = toolkit_path / "run.py"

        # Use the specific venv python
        venv_python = Path("C:/Apps/sd-scripts/venv/Scripts/python.exe")
        if venv_python.exists():
            python_exe = str(venv_python)
        else:
            python_exe = sys.executable

        command = f'"{python_exe}" "{run_script}" "{yaml_path}"'

        return command, None

