#!/bin/bash

# Script to clean data directories for SemDeDup project

# Define the base data directory
DATA_DIR="data"

# Define default subdirectories to clean
DEFAULT_DIRS=(
  "dataframes"
  "embeddings"
  "clustering"
  "sorted_clusters"
  "statistics/dataframes"
  "statistics/dicts"
)

# Initialize arrays
DIRS_TO_CLEAN=()
DIRS_TO_EXCLUDE=()

# Print usage information
function print_usage {
  echo "Usage: $0 [OPTIONS]"
  echo
  echo "Options:"
  echo "  -h, --help              Display this help message"
  echo "  -c, --clean FOLDER      Specify folder(s) to clean (can be used multiple times)"
  echo "  -e, --exclude FOLDER    Specify folder(s) to exclude from cleaning (can be used multiple times)"
  echo "  -a, --all               Clean all default folders"
  echo
  echo "Default folders:"
  for dir in "${DEFAULT_DIRS[@]}"; do
    echo "  - ${DATA_DIR}/${dir}"
  done
  echo
  echo "If no folders are specified with -c/--clean, all default folders will be cleaned"
  echo "except those explicitly excluded with -e/--exclude."
}

# Parse command line arguments
CLEAN_ALL=true
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      print_usage
      exit 0
      ;;
    -c|--clean)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --clean requires a folder name"
        exit 1
      fi
      DIRS_TO_CLEAN+=("$2")
      CLEAN_ALL=false
      shift 2
      ;;
    -e|--exclude)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --exclude requires a folder name"
        exit 1
      fi
      DIRS_TO_EXCLUDE+=("$2")
      shift 2
      ;;
    -a|--all)
      CLEAN_ALL=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# If no specific folders were provided, use the default list
if [ ${#DIRS_TO_CLEAN[@]} -eq 0 ] && [ "$CLEAN_ALL" = true ]; then
  for dir in "${DEFAULT_DIRS[@]}"; do
    DIRS_TO_CLEAN+=("$dir")
  done
fi

# Process the directories
echo "Starting cleanup of SemDeDup data directories..."

# Function to check if a directory is in the exclude list
is_excluded() {
  local check_dir="$1"
  for exclude_dir in "${DIRS_TO_EXCLUDE[@]}"; do
    # Remove data/ prefix if present for comparison
    local clean_exclude="${exclude_dir#${DATA_DIR}/}"
    if [[ "$check_dir" == "$clean_exclude" || "$check_dir" == *"$clean_exclude"* ]]; then
      return 0  # True, directory is excluded
    fi
  done
  return 1  # False, directory is not excluded
}

# Function to normalize directory path
normalize_path() {
  local dir="$1"
  # Remove data/ prefix if present
  local clean_dir="${dir#${DATA_DIR}/}"
  # Return full path with data/ prefix
  echo "${DATA_DIR}/${clean_dir}"
}

# Clean directories
for dir in "${DIRS_TO_CLEAN[@]}"; do
  # Normalize the path to ensure it's under DATA_DIR
  full_dir=$(normalize_path "$dir")
  
  # Strip the directory for exclusion checking
  dir_to_check="${dir#${DATA_DIR}/}"
  
  # Check if the directory should be excluded
  if is_excluded "$dir_to_check"; then
    echo "Skipping excluded directory: $full_dir"
    continue
  fi

  if [ -d "$full_dir" ]; then
    echo "Cleaning directory: $full_dir"
    rm -rf "$full_dir"/*
    echo "✓ Directory cleaned: $full_dir"
  else
    echo "Directory does not exist, creating: $full_dir"
    mkdir -p "$full_dir"
    echo "✓ Directory created: $full_dir"
  fi
done

echo "Cleanup complete!" 