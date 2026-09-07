import argparse
from typing import override

import ollama

from .group_abc import Group, GroupArgs
from .json_loads_type import json_loads_dict


class LanguageProcessingArgs(GroupArgs):
    ollama_model: str
    ollama_options: ollama.Options


class LanguageProcessingGroup(Group[LanguageProcessingArgs]):
    """Group of arguments related to natural language processing."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Language Processing", "Extracting event announcements from text")
        group.add_argument(
            "--ollama-model",
            default="qwen2.5:7b-instruct-q6_K",
            help="The model to use for processing text (default: %(default)s)",
        )
        group.add_argument(
            "--ollama-options",
            type=lambda s: ollama.Options(**json_loads_dict(s)),
            default={
                "temperature": 0.0,
                "num_predict": 512,
                "num_ctx": 2048,
                "repeat_penalty": 1.05,
            },
            help="A JSON dictionary of keyword arguments to pass to ollama.Options (default: %(default)s)",
        )

    @override
    def resolve(self, args: argparse.Namespace) -> LanguageProcessingArgs:
        return LanguageProcessingArgs(
            ollama_model=args.ollama_model,
            ollama_options=args.ollama_options,
        )
