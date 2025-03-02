import sys
import argparse
from pathlib import Path

from .dataset import CocoDataset


def main():
    parser = argparse.ArgumentParser(description="sane-coco: Better tools for COCO datasets")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    info_parser = subparsers.add_parser("info", help="Display dataset information")
    info_parser.add_argument("annotation_file", type=str, help="Path to COCO annotation file")
    
    filter_parser = subparsers.add_parser("filter", help="Filter a dataset")
    filter_parser.add_argument("annotation_file", type=str, help="Path to COCO annotation file")
    filter_parser.add_argument("--output", "-o", type=str, required=True, help="Output annotation file")
    filter_parser.add_argument("--categories", type=str, help="Comma-separated list of category names")
    filter_parser.add_argument("--min-area", type=float, help="Minimum annotation area")
    filter_parser.add_argument("--max-area", type=float, help="Maximum annotation area")
    filter_parser.add_argument("--no-crowds", action="store_true", help="Exclude crowd annotations")
    
    args = parser.parse_args()
    
    if args.command == "info":
        dataset = CocoDataset(args.annotation_file)
        print(f"Dataset: {Path(args.annotation_file).name}")
        print(f"Images: {len(dataset.images)}")
        print(f"Annotations: {len(dataset.annotations)}")
        print(f"Categories: {len(dataset.categories)}")
        print("\nCategory counts:")
        for cat in dataset.categories:
            ann_count = len(cat.annotations)
            print(f"  {cat.name}: {ann_count} annotations")
    
    elif args.command == "filter":
        print("Filter functionality not yet implemented")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())