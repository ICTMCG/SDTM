#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_PATH="samples/demos"

CAPTIONS=(
	"There is a piece of cake on a plate with decorations on it"
	"A bathroom with a bathtub and a sink overlooking a blue ocean"
	"A cat sitting on a street corner looking at the camera"
	"A glass bowl filled with oranges on a table"
	"an airplane sitting on the tarmac with clouds above it"
	"An orange reddish rose in a vase filled with water on top of a table"
	"From the rear, we see a horse standing between two cars parked in a parking lot"
	"The tiny kitten looks sad sitting in the bathroom sink with a scrub brush on the rim"
)

echo "Running ${#CAPTIONS[@]} captions sequentially..."
for caption in "${CAPTIONS[@]}"; do
	echo "[RUN] ${caption}"
	python "${DIR}/sample_demo.py" \
		--caption "${caption}" \
		--output-path "${OUTPUT_PATH}"
done

echo "All done. Outputs under: ${DIR}/${OUTPUT_PATH}"