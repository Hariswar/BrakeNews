# Vehicle illustration: 3D decision

Keep the schematic for this release. The source data maps complaint themes to broad systems,
not to a precise part or location on an individual vehicle. An exterior model that rotates would
add visual interest, but there is no evidence yet that it would make these reports easier to
understand. Engine, electrical and steering issues would still require the text explanations.

No GLB, glTF, FBX or OBJ asset is present in this repository, and there is no associated vehicle
model licence or attribution. No 3D model was downloaded or added during this review.

## Asset needed for a later pilot

Use the **2025 Honda Pilot** page as a single-page pilot, alongside the existing schematic:

- A web-ready GLB accurately representing the 2025 Honda Pilot, with model-year/generation
  compatibility documented by its source. If only a generic vehicle is available, label it
  clearly as illustrative and do not present it as the selected car's exact design.
- An identifiable creator, original source URL, retained licence text and required attribution.
  The licence must allow the intended public/commercial website display, redistribution of
  the downloaded model, and any optimisation or edits. Any restrictions on logos, textures
  and other embedded assets must also be resolved.
- A static poster image covered by the same permissions, with text alternatives. Prefer a
  compact model (a 2 MB transfer target for the pilot), modest textures and no unnecessary
  interior geometry. This is a performance budget, not evidence that an asset exists.
- Broad system markers must describe reported complaint categories. The asset must not
  invent component-level precision, identify a defective part, or depict a confirmed defect.

## Pilot behaviour and decision

Load both the viewer code and model only after an explicit “Explore in 3D” action. Preserve the
poster, schematic, text controls and all complaint evidence when loading fails or 3D is
unavailable. Provide keyboard controls, visible focus, a reset view, and useful text labels.
Keep automatic rotation off; honour reduced-motion preferences for camera transitions.

Before expanding, compare the pilot with the schematic on phones and desktops: can visitors
find a named system, open its supporting complaints, and distinguish a flag from a recall?
Measure completion and errors as well as loading, responsiveness and transfer size. Expand
only if 3D improves those tasks enough to justify sourcing and maintaining accurate models.
An attractive rotating exterior alone is insufficient evidence for expansion.
