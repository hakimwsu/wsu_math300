# Prepared video frames

`prepared_frames.npy` contains the prepared frame matrix used for the published numerical results.

| Property | Value |
| --- | --- |
| Source | Hakim Nizam's personal push-up recording, supplied November 13, 2026 |
| Original filename | `20261113_222244000_iOS.MOV` |
| Decoded frames | 184 |
| Selected frames | 92 |
| Selection | Every other decoded frame, starting at zero |
| Prepared image width and height | 54 × 96 pixels |
| Matrix shape | `(5184, 92)` |
| Data type | `uint8` |
| Pixel values | Grayscale values on the 0–255 scale |
| Mean subtraction | None |

Each matrix column stores one prepared image in NumPy's default C order: pixels across the first row, then pixels across the next row. For a zero-based column index `j`, the source is decoded frame `2*j`. Reshaping that column to `(96, 54)` restores its image layout.

The preparation uses ImageIO's FFmpeg reader and Pillow's grayscale conversion and resize operation, as implemented in `work/practice.py`. The container metadata listed 194 source frames, while the actual decoder returned 184. The experiment and this matrix use the decoded count.

The stored pixel payload occupies 476,928 bytes. The `.npy` file occupies 477,056 bytes including its format header. Its SHA-256 checksum is `1301121ae2e64c455ccd6e3df7bfa6cebdd3020e34a555395eded565943f8f16`.

The analysis uses floating-point factorizations, and its dense-float64 storage comparison is 3,815,424 bytes.

The default run loads this array with `numpy.load(..., allow_pickle=False)` and reproduces the recorded experiment. An optional local video path runs the preparation steps again. The source recording is retained locally by Hakim. Publication of these prepared frames accompanies the report and tested code he authorized for this project.
