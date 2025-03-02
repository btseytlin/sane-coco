# CLAUDE.md - Sane-Coco Development Guide

pycocotools is an outdated python library for working with the COCO dataset formats. Despite being developed literally 11 years ago it's still widely used in the community. 

The goal of this project is to provide a modern Python library for working with the COCO dataset formats.

Core features:
- Pure python as much as possible, type hints, tests, 2025 code style and practices.
- No deep abstraction nesting. Simple pythonic API. At most 2 levels of class inheritance. 
- Reading and writing COCO JSON files.
- Providing python data structures: easily searching, filtering, and manipulating annotations.
- Using COCO format datasets in pytorch.
- Performing evaluation just like the coco evaluator.
- NO COMMENTS. EVER.

## Project Structure
- `cocoapi/PythonAPI/pycocotools` - legacy Python wrapper for COCO API
- `sane_coco` - Python wrapper for COCO API under development
- `tests` - unit tests for `sane_coco`

## Code Style Guidelines
- Follow PEP 8.
- No docstrings or comments. NEVER USE ANY COMMENTS.
- Use type hints.
- Use f-strings.
- Sort imports: standard library → third-party → local
- Use type annotations for all function parameters and return values.
- Error handling: Use specific exceptions with descriptive messages.
- Keep functions small and focused on a single responsibility.
- Use `dataclasses` extensively.
- Use pytest for tests.
- Avoid using private attributes and methods, so no "_" prefixes.
- Use Pillow for images
- Dont use opencv.
- NO COMMENTS. EVER.