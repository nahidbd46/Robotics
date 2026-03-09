# Robotics
Differential Drive Robot Simulation using Python.

## Melt Pool & Spatter Analysis

The `melt_pool_analysis` package provides tools for additive-manufacturing
process monitoring. It can **detect**, **classify**, and **regenerate**
melt pools and spatters in image sequences.

### Features

- **Detection** – locate melt-pool and spatter regions using classical
  image-processing (thresholding, morphological ops, contour analysis).
- **Classification** – classify cropped regions as *normal melt pool*,
  *abnormal melt pool*, or *spatter* with a lightweight CNN.
- **Regeneration** – reconstruct missing melt pools via temporal
  interpolation or OpenCV inpainting.
- **Pipeline** – an end-to-end entry point combining all three stages.

### Quick start

```bash
pip install -r requirements.txt
```

```python
import cv2
from melt_pool_analysis import MeltPoolPipeline

pipeline = MeltPoolPipeline()

# Analyse a single frame
frame = cv2.imread("frame_001.png")
result = pipeline.analyse_frame(frame)
print(result["has_melt_pool"], result["detections"])

# Process a sequence and regenerate missing melt pools
frames = [cv2.imread(f"frame_{i:03d}.png") for i in range(10)]
results = pipeline.analyse_sequence(frames, regenerate_missing=True)
print(MeltPoolPipeline.summary(results))
```

### Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
