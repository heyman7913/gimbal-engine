# The mesh model: multi homography stabilization

A single homography assumes the whole scene moves as one plane. That holds for camera rotation and
distant content, but it breaks on parallax, where near and far objects move by different amounts,
and on any motion that is local to part of the frame. The mesh model replaces the one global
homography with a grid of local ones, so different regions of the frame can be corrected
differently.

## The model

`MeshIHN` builds on the global IHN. It runs the global model first to capture the dominant motion,
then a small head predicts a per cell residual on top. The output is a set of four point offsets
for each cell of an 8x8 grid, and each cell's offsets are solved to a homography by the same
differentiable Tensor-DLT the global model uses. `mesh_sampling_grid` then blends the per cell
homographies into one smooth sampling field: each pixel interpolates the homographies of the four
cells around its position, so the field bends continuously rather than jumping at cell borders.

Because the residual is added on top of the global motion, a 1x1 grid is identical to the global
IHN. This is the property that keeps the mesh model honest: it can only help, because at the
coarsest grid it reproduces the global model exactly.

## Smoothness

Left unconstrained, neighbouring cells can disagree and tear the warp field. `mesh_smoothness`
penalises the difference between neighbouring cells' offsets, which is the inductive bias that a
real parallax field is locally smooth. The weight on this penalty trades local flexibility against
field coherence, and the sweep below shows its effect.

## Results

These are measured on an RTX 5070 Ti laptop GPU with a short training run on synthetic data, where
the ground truth motion field is known. They are the foundation demonstration for the model, not a
full scale training result.

### Equivalence at a 1x1 grid

The mesh sampling field at a 1x1 grid matches the single homography warp to 1e-5, so the mesh model
provably reduces to the global model. This is covered by a test.

### Parallax against global motion

On synthetic parallax (a smoothly varying per cell translation field that no single homography can
represent), the mesh model fits the local motion the global model cannot. The figures below are at
a smoothness weight of 0.1.

| Model | MACE on synthetic parallax |
|---|---|
| Global homography | 11.18 px |
| **Mesh, 8x8 grid** | **4.69 px** |

On a single global motion the picture is the opposite. The extra per cell capacity, trained here
only on parallax, does not collapse back to one homography on its own: the per cell offsets vary by
about 5 px, and the mesh trails the global model (17.05 px against 15.96 px). Making the trained
grid reduce to global on global motion, the way the 1x1 grid does by construction, is what mixing
global motion into training is for.

### Smoothness sweep

The smoothness penalty pulls neighbouring cells together. Raising it lowers the per cell
disagreement on global motion, at a small cost to the parallax gain.

| Smoothness weight | Mesh gain on parallax (px) | Cell offset spread on global motion (px) |
|---|---|---|
| 0.0 | 6.55 | 5.34 |
| 0.1 | 6.49 | 5.03 |
| 0.5 | 6.09 | 4.46 |

## Scope

This is the foundation: the model, the equivalence guarantee, and the synthetic results that show
the grid bending to local motion. Training the mesh model on real footage and checking it does not
regress the global result on the NUS categories is the next step.
