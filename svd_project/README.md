# Video compression with an incremental SVD

This MATH 300 project compares a batch singular value decomposition with an incremental column-update SVD for a personal push-up video. The prepared data contain 92 grayscale frames, each 54 pixels wide and 96 pixels high. Flattened frames form a matrix of shape `(5184, 92)`.

The algorithm processes blocks of ten frames, ending with a two-frame block. An untruncated run checks agreement with the batch SVD. Separate rank-1, rank-5 and rank-20 runs compare reconstruction errors and visible detail. The expanded basis is reorthogonalized with QR, and its triangular factor is carried into the small SVD to preserve the represented matrix.

## Install and run

From the root of the cloned repository, use the following commands in PowerShell. The recorded experiment used Python 3.14.6. The package versions are pinned in `requirements.txt`.

```powershell
cd svd_project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe work/practice.py --no-show
```

On Linux or macOS, the corresponding commands are as follows.

```sh
cd svd_project
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python work/practice.py --no-show
```

The default run loads `data/prepared_frames.npy`, so the prepared experiment is reproducible from the repository. The data are described in [data/README.md](data/README.md). Running without `--no-show` also opens the plots.

To prepare frames from a local video instead, supply its path as the first argument.

```powershell
.\.venv\Scripts\python.exe work/practice.py "C:\path\to\video.MOV" --no-show
```

Video input uses ImageIO's FFmpeg plugin. The loader keeps every other decoded frame from index zero, stops after 100 selected frames, converts each image to grayscale, and resizes it to width 54 and height 96. The original recording is retained locally by its owner.

Run the commands from the `svd_project` directory. The script saves or replaces these generated results in `outputs/`.

- `singular_values.png` compares batch and untruncated incremental spectra on a logarithmic vertical axis.
- `error_vs_rank.png` compares whole-matrix relative Frobenius errors at the three retained ranks.
- `frame_reconstructions.png` shows the same original frame and its batch and incremental reconstructions.
- `rank_comparison.csv` records errors and factor-storage estimates.

## Expected results

The included data produce `Y.shape == (5184, 92)`. The small validation matrix has singular values `5, 5, 2, 0` and is reconstructed within floating-point tolerance. The recorded full-video run had an untruncated relative reconstruction error of about `8.05e-15`, relative spectrum error of `8.84e-16`, and left-basis orthogonality error of `1.56e-14`. Last digits can vary across numerical libraries and machines.

| Retained rank | Batch relative error | Incremental relative error |
| --- | --- | --- |
| 1 | 0.272091762467 | 0.272635800540 |
| 5 | 0.0962022454131 | 0.0968104477814 |
| 20 | 0.0412843540311 | 0.0416514004080 |

Errors use the floating-point reconstruction before display clipping. The image comparison uses sampled frame 47 of 92 and a shared grayscale display range of 0–255. The program preloads the matrix and constructs full reconstructions. Its factor-byte counts describe stored arrays. They do not measure peak memory, runtime improvements, or encoded-video compression ratios.

## Report

[The LaTeX source](report/svd_project.tex) retains the wording of Hakim's latest supplied report. [The report PDF](report/svd_project.pdf) is rebuilt from that source so the published source and PDF match. The separate [AI-use record](AI_USE_RECORD.txt) documents the assistance that occurred.

With a LaTeX installation available, compile twice from the `svd_project` directory to resolve references.

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/svd_project.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/svd_project.tex
```

The supplied `outputs/report_spectrum.png`, `outputs/report_error.png`, and `outputs/report_frames.png` are the figure assets used by this report. To regenerate those compact report figures from the prepared data, run the figure builder, then compile the report twice as above.

```powershell
.\.venv\Scripts\python.exe work/report_build/build_report_figures.py
```

The main analysis script exports the larger comparison figures listed above. [metadata.json](data/metadata.json) records the prepared data and experiment details. [The code-to-source map](references/CODE_AND_SOURCE_MAP.txt) and [verified-source notes](references/VERIFIED_SOURCES.txt) connect the implementation and report to the references.

## Files

```text
svd_project/
  README.md
  AI_USE_RECORD.txt
  requirements.txt
  data/
    README.md
    metadata.json
    prepared_frames.npy
  work/
    practice.py
    report_build/
      build_report_figures.py
  references/
    CODE_AND_SOURCE_MAP.txt
    VERIFIED_SOURCES.txt
  outputs/
    rank_comparison.csv
    *.png
  report/
    svd_project.tex
    svd_project.pdf
```

## Class notes and further sources

The dates below identify the lessons used. The links pin the referenced course files to specific commits.

- September 18, 2026 covers array shapes, reshaping, and matrix multiplication used to assemble and interpret the frame matrix. [NumPy class notebook](https://github.com/say-yas/wsu_math300/blob/8acbae688a5e5ef588dca37c5ca6f17b00600ac4/Sep18/class_intro_numpy_2.ipynb)
- September 25, 2026 covers floating-point error and tolerances used to interpret the validation checks. [Floating-point and convergence practice](https://github.com/say-yas/wsu_math300/blob/2e41141c9a3646473619b922458d42a1e5333f66/Sep25/complexity_fp_convergence_practice.ipynb)
- November 6, 2026 covers reduced QR, projection, and checking basis orthogonality. [Matrix decompositions, Part B](https://github.com/say-yas/wsu_math300/blob/ef5f38f850dabb9a1f19068884ba4b0fa7bb2a5b/Nov6_decompositions/decompositions.ipynb)
- November 9, 2026 covers reduced SVD factor shapes and NumPy's already-transposed `Vt` convention. [SVD and eigenspaces, Part A](https://github.com/say-yas/wsu_math300/blob/435aab704ac2701d771769e232c75295ec97bba9/Nov9_SVD_similarity_transformations/svd-eigenspaces.ipynb)
- November 13, 2026 covers rank-based image reconstruction and factor-storage estimates. [Image compression with the SVD](https://github.com/say-yas/wsu_math300/blob/2242bf2f92da2d12b0ac8d933342b008428daaa1/Nov13_Imagecompression/svd_image_compression.ipynb)

The shared Overleaf activities XXX, XXXI and XXXII were also consulted. [WSU_MATH300_Shared](https://www.overleaf.com/project/6a8b1c59a232e8dbbb5b0923)

The incremental update follows Kühl et al. (2023), Section 2.2, equations 16–19. Brand (2006) develops related thin-SVD modification formulas. Damle's Cornell notes, pages 2–3, explain the rank-one expansion and optimal batch error. Zhang (2022), Section 2, discusses numerical orthogonality in incremental SVD. These sources support the methods and their interpretation. [Kühl et al.](https://arxiv.org/abs/2302.09149v1), [Brand](https://www.merl.com/publications/TR2006-059), [Damle](https://www.cs.cornell.edu/courses/cs3220/2020fa/SVD.pdf), [Zhang, version 2](https://arxiv.org/abs/2204.05398v2)

The report bibliography also links the official NumPy, ImageIO, Pillow and Matplotlib documentation for the operations used.

## Contributions and AI use

Hakim Nizam identified himself as the sole student participant. He supplied the recording, entered and ran early examples, reported outputs, reviewed results and figures, edited and supplied the latest report, and authorized publication. Group contribution percentages have not been established.

OpenAI Codex supplied and edited substantial Python code for preprocessing, batch and incremental SVD, reorthogonalization, truncation, validation, experiments and plotting. Codex generated the original report draft and its figures and layout, revised explanations, checked sources, and prepared the portable data-loading and repository documentation. The latest report supplied by Hakim includes his subsequent edits. The project is presented with this record of actual assistance. Citations identify mathematical and software sources, while this statement identifies the recorded authorship assistance. Further detail appears in [AI_USE_RECORD.txt](AI_USE_RECORD.txt).
