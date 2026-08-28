#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
work_dir=$(mktemp -d /tmp/tvn-map-tiles.XXXXXX)
trap 'rm -rf "$work_dir"' EXIT

tile_size=1024
quality=90

generate_tiles() {
  local key=$1
  local pdf=$2
  local dpi=$3
  local width=$4
  local height=$5
  local output_dir="$repo_dir/maps/previews/$key-hq"
  local raster="$work_dir/$key.jpg"

  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing tile directory: $output_dir" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  pdftoppm -f 1 -l 1 -singlefile -r "$dpi" -jpeg -jpegopt quality=92,progressive=y \
    "$repo_dir/$pdf" "$work_dir/$key"

  local y=0
  while (( y < height )); do
    local x=0
    local crop_height=$((height - y))
    (( crop_height > tile_size )) && crop_height=$tile_size
    while (( x < width )); do
      local crop_width=$((width - x))
      (( crop_width > tile_size )) && crop_width=$tile_size
      cwebp -quiet -mt -q "$quality" -sharp_yuv \
        -crop "$x" "$y" "$crop_width" "$crop_height" \
        "$raster" -o "$output_dir/$((x / tile_size))-$((y / tile_size)).webp"
      x=$((x + tile_size))
    done
    y=$((y + tile_size))
  done
}

generate_tiles vfr "maps/00_VFRspredaj_25_SC_WEB_flat.pdf" 254 10102 7002
generate_tiles legacy "maps/1 200k original karta.pdf" 203 3357 2374
generate_tiles p250 "maps/P25004BR.pdf" 72 11890 8410

echo "High-resolution preview tiles generated under maps/previews/."
