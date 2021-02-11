# Process
1. Download human3.6m
2. Use `event_library` generator script to generate `raw` events from `mp4` files:
```
python event_library/tools/generate.py frames_dir=path/to/dataset put_dir=out upsample=true emulate=true search=false representation=raw
```
3. Launch `scripts/generate_data.py` to generate `constant_count` frames and joints
