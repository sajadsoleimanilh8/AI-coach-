"""The GGUF export instructions users are told to run."""

from __future__ import annotations

import re

import pytest

from nexus.training.quantize import (
    CONVERTER_OUTTYPES,
    INTERMEDIATE_OUTTYPE,
    gguf_conversion_instructions,
    ollama_create_command,
    write_modelfile,
)

_OUTTYPE_RE = re.compile(r"--outtype\s+(\S+)")


def _instructions(quantization: str = "q4_k_m") -> str:
    return gguf_conversion_instructions("out/merged", "out/nexus-custom-q4_k_m.gguf", quantization)


def test_every_printed_outtype_is_one_the_converter_accepts() -> None:
    emitted = _OUTTYPE_RE.findall(_instructions())

    assert emitted, "instructions must actually contain a --outtype to convert with"
    for outtype in emitted:
        assert outtype in CONVERTER_OUTTYPES, (
            f"--outtype {outtype} is not accepted by convert_hf_to_gguf.py; "
            f"it only writes {CONVERTER_OUTTYPES}"
        )


@pytest.mark.parametrize("quantization", ["q4_k_m", "q5_k_m", "q6_k", "q8_0"])
def test_a_k_quant_is_never_passed_to_the_converter(quantization: str) -> None:
    text = _instructions(quantization)

    assert quantization not in _OUTTYPE_RE.findall(text)
    assert quantization.upper() in text


def test_both_commands_are_emitted_in_the_order_they_must_be_run() -> None:
    text = _instructions()

    convert_at = text.index("convert_hf_to_gguf.py")
    quantize_at = text.index("llama-quantize")
    assert convert_at < quantize_at, "converting must come before quantizing"


def test_the_intermediate_file_is_named_explicitly_and_reused() -> None:
    text = gguf_conversion_instructions("out/merged", "out/nexus-custom-q4_k_m.gguf", "q4_k_m")
    expected_intermediate = f"nexus-custom-q4_k_m-{INTERMEDIATE_OUTTYPE}.gguf"

    assert text.count(expected_intermediate) >= 2
    quantize_line = next(line for line in text.splitlines() if "llama-quantize" in line)
    assert expected_intermediate in quantize_line
    assert "nexus-custom-q4_k_m.gguf" in quantize_line


def test_the_intermediate_outtype_is_itself_a_valid_converter_outtype() -> None:
    assert INTERMEDIATE_OUTTYPE in CONVERTER_OUTTYPES


def test_the_final_target_is_the_quantized_file_not_the_intermediate() -> None:
    text = gguf_conversion_instructions("merged", "output/model-q4_k_m.gguf", "q4_k_m")

    quantize_line = next(line for line in text.splitlines() if "llama-quantize" in line)
    parts = quantize_line.split()
    assert parts[-1] == "Q4_K_M"
    assert parts[-2].endswith("model-q4_k_m.gguf")
    assert parts[-3].endswith(f"model-q4_k_m-{INTERMEDIATE_OUTTYPE}.gguf")


def test_modelfile_points_at_the_gguf_and_is_registrable(tmp_path) -> None:
    gguf = tmp_path / "nexus-custom-q4_k_m.gguf"
    modelfile = write_modelfile(gguf_path=gguf, output_path=tmp_path / "Modelfile")

    content = modelfile.read_text(encoding="utf-8")
    assert content.startswith(f"FROM {gguf.as_posix()}")
    assert "PARAMETER num_ctx 8192" in content
    assert ollama_create_command("nexus-custom", modelfile).startswith("ollama create nexus-custom -f ")
