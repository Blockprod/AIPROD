"""
HuggingFace Hub Utilities — Cloud upload.
==========================================

Push trained LoRA weights and model cards to HuggingFace Hub.

This module lives in ``aiprod-cloud`` and is re-exported by the
backward-compatible shim at ``aiprod_trainer.hf_hub_utils``.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Union

import imageio
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)
from rich.progress import Progress, SpinnerColumn, TextColumn

from aiprod_trainer import logger
from aiprod_trainer.config import AIPRODTrainerConfig


def push_to_hub(weights_path: Path, sampled_videos_paths: List[Path], config: AIPRODTrainerConfig) -> None:
    """Push the trained LoRA weights to HuggingFace Hub."""
    if not config.hub.hub_model_id:
        logger.warning("⚠️ HuggingFace hub_model_id not specified, skipping push to hub")
        return

    api = HfApi()

    original_progress_state = are_progress_bars_disabled()
    disable_progress_bars()

    try:
        try:
            repo = create_repo(
                repo_id=config.hub.hub_model_id,
                repo_type="model",
                exist_ok=True,
            )
            repo_id = repo.repo_id
            logger.info(f"🤗 Successfully created HuggingFace model repository at: {repo.url}")
        except Exception as e:
            logger.error(f"❌ Failed to create HuggingFace model repository: {e}")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                try:
                    task_copy = progress.add_task("Copying weights...", total=None)
                    weights_dest = temp_path / weights_path.name
                    shutil.copy2(weights_path, weights_dest)
                    progress.update(task_copy, description="✓ Weights copied")

                    task_card = progress.add_task("Creating model card and samples...", total=None)
                    _create_model_card(
                        output_dir=temp_path,
                        videos=sampled_videos_paths,
                        config=config,
                    )
                    progress.update(task_card, description="✓ Model card and samples created")

                    task_upload = progress.add_task("Pushing files to HuggingFace Hub...", total=None)
                    api.upload_folder(
                        folder_path=str(temp_path),
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    progress.update(task_upload, description="✓ Files pushed to HuggingFace Hub")
                    logger.info("✅ Successfully pushed files to HuggingFace Hub")

                except Exception as e:
                    logger.error(f"❌ Failed to process and push files to HuggingFace Hub: {e}")
                    raise

    finally:
        if not original_progress_state:
            enable_progress_bars()


def convert_video_to_gif(video_path: Path, output_path: Path) -> None:
    """Convert a video file to GIF format."""
    try:
        reader = imageio.get_reader(str(video_path))
        fps = reader.get_meta_data()["fps"]

        writer = imageio.get_writer(
            str(output_path),
            fps=min(fps, 15),
            loop=0,
        )

        for frame in reader:
            writer.append_data(frame)

        writer.close()
        reader.close()
    except Exception as e:
        logger.error(f"Failed to convert video to GIF: {e}")


def _create_model_card(
    output_dir: Union[str, Path],
    videos: List[Path],
    config: AIPRODTrainerConfig,
) -> Path:
    """Generate and save a model card for the trained model."""

    repo_id = config.hub.hub_model_id
    pretrained_model_name_or_path = config.model.model_path
    validation_prompts = config.validation.prompts
    output_dir = Path(output_dir)
    template_path = Path(__file__).parent.parent.parent / "templates" / "model_card.md"

    template = template_path.read_text()

    model_name = repo_id.split("/")[-1]

    base_model_link = str(pretrained_model_name_or_path)
    model_path_str = str(pretrained_model_name_or_path)
    is_url = model_path_str.startswith(("http://", "https://"))

    base_model_name = model_path_str.split("/")[-1] if is_url else Path(pretrained_model_name_or_path).name

    prompts_text = ""
    sample_grid = []

    if validation_prompts and videos:
        prompts_text = "Example prompts used during validation:\n\n"

        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True, parents=True)

        cells = []
        for i, (prompt, video) in enumerate(zip(validation_prompts, videos, strict=False)):
            if video.exists():
                prompts_text += f"- `{prompt}`\n"

                gif_path = samples_dir / f"sample_{i}.gif"
                try:
                    convert_video_to_gif(video, gif_path)

                    cell = (
                        f"![example{i + 1}](./samples/sample_{i}.gif)"
                        "<br>"
                        '<details style="max-width: 300px; margin: auto;">'
                        f"<summary>Prompt</summary>"
                        f"{prompt}"
                        "</details>"
                    )
                    cells.append(cell)
                except Exception as e:
                    logger.error(f"Failed to process video {video}: {e}")

        num_cells = len(cells)
        if num_cells > 0:
            num_cols = min(4, num_cells)
            num_rows = (num_cells + num_cols - 1) // num_cols

            for row in range(num_rows):
                start_idx = row * num_cols
                end_idx = min(start_idx + num_cols, num_cells)
                row_cells = cells[start_idx:end_idx]
                formatted_row = "| " + " | ".join(row_cells) + " |"
                sample_grid.append(formatted_row)

    grid_text = "\n".join(sample_grid) if sample_grid else ""

    model_card_content = template.format(
        base_model=base_model_name,
        base_model_link=base_model_link,
        model_name=model_name,
        training_type="LoRA fine-tuning" if config.model.training_mode == "lora" else "Full model fine-tuning",
        training_steps=config.optimization.steps,
        learning_rate=config.optimization.learning_rate,
        batch_size=config.optimization.batch_size,
        validation_prompts=prompts_text,
        sample_grid=grid_text,
    )

    model_card_path = output_dir / "README.md"
    model_card_path.write_text(model_card_content)

    return model_card_path
