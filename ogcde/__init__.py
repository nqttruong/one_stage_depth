"""OGCDE one-stage: YOLO + geometry multi-task head.

Modules:
  model    - Backbone / neck / extended detection head
  loss     - Multi-task loss (det + depth + scale + contact + geo)
  dataset  - KITTI loader with contact point / depth / distance GT
  metrics  - AbsRel, RMSE, delta, DE, CPE, Stability
  utils    - Letterbox, NMS, decoding, calibration helpers
"""
__version__ = "0.1.0"
