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
- Drop the filtering functionality. 
- Drop the cmdline interface.


## Project Structure
- `cocoapi/PythonAPI/pycocotools` - legacy Python wrapper for COCO API
- `sane_coco` - Python wrapper for COCO API under development
- `tests` - unit tests for `sane_coco`

## Design Principles
- Intuitive object relationships (images have annotations, not just IDs)
- Native Python iteration (for annotation in dataset.annotations)
- Pythonic data structures (BBox, Mask, RLE) over faceless dicts, tuples and arrays
- Context-preserving views instead of copies
- Type hints everywhere, IDE-friendly autocomplete
- Conversion utilities (RLE↔polygon↔mask) with sensible defaults
- Visualization helpers that work with matplotlib/notebook ecosystem
- Serialization that preserves custom attributes
- Short and simple. No Comments! No docstrings! 
- No magic!

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
- Modern Python idioms and design patterns
- Direct object references instead of ID-based lookups
- Dataclasses for clean type definitions
- Python-native filtering with list/dict comprehensions.
- Properties instead of methods where appropriate.
- NumPy integration for mask operations.

