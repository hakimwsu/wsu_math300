import imageio.v3 as iio
from pathlib import Path
import sys

video_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "--no-show" else None

from PIL import Image
import numpy as np

frames = iio.imiter(video_path, plugin="FFMPEG") if video_path else []
columns = []
for index, frame in enumerate(frames): 
    if index % 2 == 0:
        img = Image.fromarray(frame)
        gray = img.convert("L")
        small = gray.resize((54, 96))
        pixels = np.asarray(small)
        columns.append(pixels.ravel())
        if len(columns) == 100:
            break
if video_path: frames.close()

Y = np.column_stack(columns) if video_path else np.load(Path(__file__).resolve().parents[1] / "data" / "prepared_frames.npy", allow_pickle=False)
U, s, vt = np.linalg.svd(Y, full_matrices=False)

print(Y.shape)

A = Y[:, :10]
B = Y[:, 10:20]
print(A.shape, B.shape)

U0, s0, vt0 = np.linalg.svd(A, full_matrices=False)
M = U0.T @ B
P = B - U0 @ M
print(M.shape, P.shape)
print("Relative size of the leftover:", np.linalg.norm(P) / np.linalg.norm(B))

Q, R = np.linalg.qr(P, mode='reduced')
print(Q.shape, R.shape)
print(np.allclose(Q @ R, P))

K = np.block([
    [np.diag(s0), M],
    [np.zeros((R.shape[0], s0.size)), R]
])
Uk, sk, vtk = np.linalg.svd(K, full_matrices=False)
print("K shape:", K.shape)

s_batch_20 = np.linalg.svd(Y[:, :20], compute_uv=False)
print("Singular values match:", np.allclose(sk, s_batch_20))

U20 = np.column_stack([U0, Q]) @ Uk
old_weights = vtk[:, :len(s0)] @ vt0
new_weights = vtk[:, len(s0):] 
vt20 = np.column_stack([old_weights, new_weights])

Y20_rebuilt = (U20 * sk) @ vt20
print("Rebuilt shape:", Y20_rebuilt.shape)
print("First 20 frames match:", np.allclose(Y20_rebuilt, Y[:, :20]))

def incremental_svd(matrix, block_size=10, q=None):
    """Append blocks with QR reorthogonalization and an optional retained rank.

    q=None keeps all modes. Otherwise, truncate the first block and every update.
    This experiment receives the preloaded matrix used for batch validation.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if q is not None and (not isinstance(q, (int, np.integer)) or q < 1):
        raise ValueError("q must be None or a positive integer")

    U_step, s_step, vt_step = np.linalg.svd(
        matrix[:, :block_size], full_matrices=False
    )
    U_step, s_step, vt_step = U_step[:, :q], s_step[:q], vt_step[:q, :]

    for start in range(block_size, matrix.shape[1], block_size):
        B = matrix[:, start:start + block_size]
        M = U_step.T @ B
        P = B - U_step @ M
        Q, R = np.linalg.qr(P, mode="reduced")
        old_rank = s_step.size
        K = np.block([
            [np.diag(s_step), M],
            [np.zeros((R.shape[0], old_rank)), R],
        ])

        # Preserve the represented matrix while restoring an orthonormal basis.
        basis, basis_R = np.linalg.qr(
            np.column_stack((U_step, Q)), mode="reduced"
        )
        Uk, new_s, vtk = np.linalg.svd(basis_R @ K, full_matrices=False)
        Uk, new_s, vtk = Uk[:, :q], new_s[:q], vtk[:q, :]
        U_step = basis @ Uk
        old_weights = vtk[:, :old_rank] @ vt_step
        new_weights = vtk[:, old_rank:]
        vt_step = np.column_stack((old_weights, new_weights))
        s_step = new_s

    return U_step, s_step, vt_step


U_inc, s_inc, vt_inc = incremental_svd(Y)

small = np.array([
    [3, 0, 4, 0, 0], 
    [0, 4, 0, 3, 0],
    [0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0],
], dtype=float)

expected = np.array([5, 5, 2, 0], dtype=float)
U_small, s_small, vt_small = incremental_svd(small, block_size=2)

print("Small singular values:", s_small)
print("Known values match:", np.allclose(s_small, expected))

small_rebuilt = (U_small * s_small) @ vt_small
print("Small reconstruction matches:", np.allclose(small_rebuilt, small))

Y_rebuilt = (U_inc * s_inc) @ vt_inc

reconstruction_error = np.linalg.norm(Y_rebuilt - Y) / np.linalg.norm(Y)
spectrum_error = np.linalg.norm(s_inc - s) / np.linalg.norm(s)
orthogonality_error = np.linalg.norm(U_inc.T @ U_inc - np.eye(len(s_inc)))

print("Full reconstruction shape:", Y_rebuilt.shape)
print(f"Relative reconstruction error: {reconstruction_error:.3e}")
print(f"Relative singular-value error: {spectrum_error:.3e}")
print(f"Left orthogonality error: {orthogonality_error:.3e}")

import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parents[1] / "outputs"
output_dir.mkdir(exist_ok=True)
indices = np.arange(1, len(s) + 1)

spectrum_figure = plt.figure()
plt.semilogy(indices, s, label="Batch SVD")
plt.semilogy(indices, s_inc, "--", label="Incremental SVD")
plt.xlabel("Singular-value index")
plt.ylabel("Singular Value")
plt.title("Singular value")
plt.legend()
plt.tight_layout()

# Each rank gets its own incremental pass, including truncation at initialization.
ranks = [1, 5, 20]
frame_index = Y.shape[1] // 2
batch_images = {}
incremental_images = {}
comparison_rows = []
video_norm = np.linalg.norm(Y)
print("\nRank    Batch error    Incremental error")

for q in ranks:
    batch_rebuilt = (U[:, :q] * s[:q]) @ vt[:q, :]
    U_q, s_q, vt_q = incremental_svd(Y, block_size=10, q=q)
    incremental_rebuilt = (U_q * s_q) @ vt_q
    batch_error = np.linalg.norm(Y - batch_rebuilt) / video_norm
    incremental_error = np.linalg.norm(Y - incremental_rebuilt) / video_norm
    tail_error = np.linalg.norm(s[q:]) / np.linalg.norm(s)
    np.testing.assert_allclose(batch_error, tail_error, rtol=1e-10, atol=1e-12)
    if incremental_error + 1e-12 < batch_error:
        raise AssertionError("Incremental error is below the batch optimum")

    factor_bytes = U_q.nbytes + s_q.nbytes + vt_q.nbytes
    dense_float64_bytes = Y.size * np.dtype(float).itemsize
    comparison_rows.append([
        q, batch_error, incremental_error, factor_bytes,
        dense_float64_bytes, Y.nbytes,
    ])
    batch_images[q] = batch_rebuilt[:, frame_index].reshape(96, 54).copy()
    incremental_images[q] = incremental_rebuilt[:, frame_index].reshape(96, 54).copy()
    print(f"{q:4d}    {batch_error:.6f}       {incremental_error:.6f}")

comparison = np.asarray(comparison_rows)
np.savetxt(
    output_dir / "rank_comparison.csv", comparison, delimiter=",", comments="",
    header="q,batch_relative_error,incremental_relative_error,factor_bytes,"
           "dense_float64_bytes,dense_uint8_bytes",
    fmt=["%d", "%.12g", "%.12g", "%d", "%d", "%d"],
)

error_figure, error_axis = plt.subplots()
error_axis.plot(ranks, comparison[:, 1], "o-", label="Batch SVD")
error_axis.plot(ranks, comparison[:, 2], "s--", label="Incremental SVD")
error_axis.set_xlabel("Retained rank q")
error_axis.set_ylabel("Relative Frobenius error")
error_axis.set_title("Whole-video error, block size 10")
error_axis.set_xticks(ranks)
error_axis.grid(alpha=0.3)
error_axis.legend()
error_figure.tight_layout()

# Shared display limits clip the view only; errors above use floating values.
frame_figure, image_axes = plt.subplots(2, 4, figsize=(11, 8))
image_axes[0, 0].imshow(Y[:, frame_index].reshape(96, 54), cmap="gray", vmin=0, vmax=255)
image_axes[0, 0].set_title("Original")
image_axes[1, 0].set_axis_off()
for column, q in enumerate(ranks, start=1):
    image_axes[0, column].imshow(batch_images[q], cmap="gray", vmin=0, vmax=255)
    image_axes[0, column].set_title(f"Batch, rank {q}")
    image_axes[1, column].imshow(incremental_images[q], cmap="gray", vmin=0, vmax=255)
    image_axes[1, column].set_title(f"Incremental, rank {q}")
for axis in image_axes.flat:
    if axis.axison:
        axis.set_xlabel("Column (pixels)")
        axis.set_ylabel("Row (pixels)")
frame_figure.suptitle(
    f"Sampled frame {frame_index + 1} of {Y.shape[1]}; shared grayscale range 0-255"
)
frame_figure.tight_layout()

for figure, filename in [
    (spectrum_figure, "singular_values.png"),
    (error_figure, "error_vs_rank.png"),
    (frame_figure, "frame_reconstructions.png"),
]:
    figure.savefig(output_dir / filename, dpi=300)
print(f"\nSaved plots and comparison data in {output_dir}")
print("Storage columns compare float64 factors with dense float64 and uint8 frames.")
print("This run preloads Y; factor storage is not a measurement of streaming peak RAM.")
if "--no-show" not in sys.argv:
    plt.show()
