from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nexus.training.config import TrainingConfig

DEFAULT_OLLAMA_MODEL_NAME = "nexus-custom"

CONVERTER_OUTTYPES = ("f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto")

INTERMEDIATE_OUTTYPE = "f16"


def gguf_conversion_instructions(
    merged_dir: str | Path, gguf_path: str | Path, quantization: str
) -> str:
    """Both commands needed to turn merged HF weights into a quantized
    GGUF, in order, with the intermediate file named explicitly.
    """
    merged = Path(merged_dir)
    target = Path(gguf_path)
    intermediate = target.with_name(f"{target.stem}-{INTERMEDIATE_OUTTYPE}.gguf")
    convert_script = Path("llama.cpp") / "convert_hf_to_gguf.py"

    return (
        f"Convert to GGUF, then quantize — these are TWO separate tools:\n"
        f"  1. python {convert_script} {merged} \\\n"
        f"       --outfile {intermediate} --outtype {INTERMEDIATE_OUTTYPE}\n"
        f"  2. llama-quantize {intermediate} {target} {quantization.upper()}\n"
        f"\n"
        f"Step 1 only writes {'/'.join(CONVERTER_OUTTYPES)}; k-quants such as "
        f"{quantization} come from llama-quantize in step 2. {intermediate.name} is an "
        f"intermediate and can be deleted once {target.name} exists."
    )

_MODELFILE_TEMPLATE = """FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER num_ctx {context_window}

SYSTEM \"\"\"You are NEXUS, a local-first assistant. State what you actually
know, separate verified facts from inference, and say plainly when you do
not know something rather than filling the gap.\"\"\"
"""


def write_modelfile(
    *,
    gguf_path: str | Path,
    output_path: str | Path,
    context_window: int = 8192,
) -> Path:
    """Writes an Ollama Modelfile pointing at the exported GGUF. Pure text
    generation with no ML imports, so it is testable on any machine."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _MODELFILE_TEMPLATE.format(
            gguf_path=Path(gguf_path).as_posix(), context_window=context_window
        ),
        encoding="utf-8",
        newline="\n",
    )
    return target


def ollama_create_command(model_name: str, modelfile_path: str | Path) -> str:
    return f"ollama create {model_name} -f {Path(modelfile_path).as_posix()}"


def merge_and_export(
    config: TrainingConfig,
    *,
    adapter_dir: str | Path,
    output_dir: str | Path,
    quantization: str = "q4_k_m",
) -> Path:
    """Merges the LoRA adapter into the base model and exports GGUF."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = Path(output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base_model, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    gguf_path = Path(output_dir) / f"nexus-custom-{quantization}.gguf"
    print(f"Merged weights written to {merged_dir}.", flush=True)
    print(gguf_conversion_instructions(merged_dir, gguf_path, quantization), flush=True)
    return gguf_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter, export GGUF, and write an Ollama Modelfile."
    )
    parser.add_argument("--adapter", type=str, required=True, help="Path to the trained adapter")
    parser.add_argument("--config", type=str, default=None, help="Path to a TrainingConfig YAML")
    parser.add_argument("--output-dir", type=str, default="nexus/training/output")
    parser.add_argument("--model-name", type=str, default=DEFAULT_OLLAMA_MODEL_NAME)
    parser.add_argument("--quantization", type=str, default="q4_k_m")
    parser.add_argument(
        "--modelfile-only",
        action="store_true",
        help="Write the Modelfile for an already-exported GGUF without merging",
    )
    parser.add_argument("--gguf", type=str, default=None, help="Existing GGUF path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = TrainingConfig.from_yaml(args.config) if args.config else TrainingConfig()
    output_dir = Path(args.output_dir)

    if args.modelfile_only:
        if not args.gguf:
            print("--modelfile-only requires --gguf")
            return 1
        gguf_path = Path(args.gguf)
    else:
        gguf_path = merge_and_export(
            config,
            adapter_dir=args.adapter,
            output_dir=output_dir,
            quantization=args.quantization,
        )

    modelfile = write_modelfile(gguf_path=gguf_path, output_path=output_dir / "Modelfile")

    print(f"\nModelfile written to {modelfile}")
    print("Register it with Ollama — it then serves through the EXISTING OllamaRuntime,")
    print("with no new provider code:\n")
    print(f"  {ollama_create_command(args.model_name, modelfile)}\n")
    print("Then uncomment the nexus-custom entry in nexus/config/models.yaml and run")
    print("  python -m nexus.training.evaluate --model-id nexus-custom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
