#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import shutil

def main():
    p = argparse.ArgumentParser(description="Pair originals and 'ours' inpaints, rename to images_00001.png/inpaints_00001.png")
    p.add_argument("input_folder", type=Path)
    p.add_argument("output_folder", type=Path)
    p.add_argument("--ext", default="png", choices=["png","jpg","jpeg"], help="Image extension to process")
    p.add_argument("--start-index", type=int, default=1, help="Starting index for numbering")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    inp = args.input_folder
    out = args.output_folder
    out.mkdir(parents=True, exist_ok=True)

    # originals look like: 13.png
    original_re = re.compile(rf"^(\d+)\.{re.escape(args.ext)}$", re.IGNORECASE)

    originals = []
    for pth in sorted(inp.glob(f"*.{args.ext}")):
        m = original_re.match(pth.name)
        if m:
            originals.append((int(m.group(1)), pth))

    idx = args.start_index
    for img_id, orig_path in originals:
        # inpaint like: 13_340_zebra_ours.png (anything ending with _ours.<ext>)
        candidates = list(inp.glob(f"{img_id}_*_ours.{args.ext}"))
        if not candidates:
            # try case-insensitive ext variations
            candidates = [p for p in inp.iterdir()
                          if p.is_file() and p.name.lower().startswith(f"{img_id}_")
                          and p.name.lower().endswith(f"_ours.{args.ext}")]
        if not candidates:
            # no pair -> skip
            continue

        inpaint_path = sorted(candidates)[0]  # pick first if multiple

        z = str(idx).zfill(5)
        dst_img = out / f"images_{z}.{args.ext}"
        dst_inp = out / f"inpaints_{z}.{args.ext}"

        if args.dry_run:
            print(f"[DRY] {orig_path} -> {dst_img}")
            print(f"[DRY] {inpaint_path} -> {dst_inp}")
        else:
            shutil.copy2(orig_path, dst_img)
            shutil.copy2(inpaint_path, dst_inp)

        idx += 1

if __name__ == "__main__":
    main()