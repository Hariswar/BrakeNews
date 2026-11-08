# BrakeNews design system

Calm, credible data journalism: warm paper, near-black ink, one safety-blue accent, serif
headlines over a plain sans body. Tokens live in `src/styles/global.css`; components use the
tokens, never raw hex.

## Color

Every text/background pair passes WCAG AA (4.5:1) in both modes; form-control borders and icons
pass 3:1. Red appears only for real do-not-drive and park-outside recall warnings.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` / `--surface` / `--surface-2` | `#f7f6f2` / `#ffffff` / `#eeece6` | `#0f1115` / `#171a20` / `#1f232b` | page, cards, panels |
| `--ink` / `--ink-2` / `--ink-3` | `#16181d` / `#454952` / `#5f636c` | `#ecedef` / `#bcc0c8` / `#9aa0aa` | text, secondary, muted |
| `--border` / `--border-strong` | `#dcd9d0` / `#8a867c` | `#2b3039` / `#6b7280` | dividers / form controls |
| `--brand` (buttons) | `#1d4e89` | `#2e68b3` | primary actions, logo |
| `--link` / `--focus` | `#1d4e89` | `#8fb6ee` | links, focus rings |
| `--conf-high` on `-soft` | `#0e5a55` on `#ddefea` | `#7fd1c3` on `#12302c` | high confidence |
| `--conf-med` on `-soft` | `#7a4e00` on `#faefd8` | `#f0c46b` on `#33270f` | medium confidence |
| `--trend` on `-soft` | `#94420e` on `#fbebdd` | `#f2a36b` on `#35210f` | rising complaints |
| `--critical` on `-soft` | `#b42318` on `#fdecea` | `#ff8a80` on `#3a1714` | recall safety warnings only |
| `--cat-*-bg` / `-fg` | 8 tints | 8 tints | category chips (text stays `--ink`) |
| `--series` | `#2a78d6` | `#3987e5` | chart marks (validated data-viz blue) |

Meaning is never carried by color alone: confidence has signal bars and a word, categories an
icon and a name, warnings an icon and a label.

## Type

- Headings: **Newsreader** (variable, 600-700), self-hosted Latin woff2, 58 KB.
- Body: **Public Sans** (variable), self-hosted Latin woff2, 27 KB.
- Both preloaded, `font-display: swap`, with measured fallbacks (Arial +3.9%, Times New Roman
  +4.3% `size-adjust`) so the swap does not shift text.
- Scale: 13 / 15 / 17 (body, line-height 1.6) / 20 / 24 / 30 / 36-48 (H1, fluid).
- `.overline`: 13px, 650, uppercase, 0.06em tracking, for vehicle names and section labels.

## Space and layout

- 8px grid: `--space-2` 8, `-3` 16, `-4` 24, `-5` 32, `-6` 48, `-7` 64, `-8` 96 (`-1` 4 for nudges).
- Container 1160px; article pages 820px; running text 70ch.
- Report grid: 1 column, 2 from 700px, 3 from 1040px.
- Radii 6 / 10 / 14px. Shadows are faint; hover lifts a card 2px.
- Motion: 180ms, `cubic-bezier(.2,.7,.2,1)`, off under `prefers-reduced-motion`.

## Components

| Component | File | Notes |
|---|---|---|
| Header | `layouts/Base.astro` | sticky, `aria-current` on the section, Menu button under 640px (links wrap without JS) |
| Footer | `components/Footer.astro` | disclaimer (wording unchanged), links, data date |
| Report card | `components/ReportCard.astro` | vehicle overline + problem title, whole card clickable via the title link, tooltips above it |
| Confidence badge | `components/ConfidenceBadge.astro` | bars + word + info tooltip |
| Category chip | `components/CategoryChip.astro` | from each problem's most common NHTSA component group (`lib/categories.ts`) |
| Trend tag | `components/TrendTag.astro` | per card only when some reports are rising and others are not |
| Info tooltip | `components/InfoTip.astro` | button + `role="tooltip"`; hover, focus or tap; Escape dismisses; kept in the viewport |
| Sparkline | `components/Sparkline.astro` | 24 months, decorative, with a screen-reader sentence |
| Vehicle lookup | `components/VehiclePicker.astro` | free-text combobox and guided dropdowns share the vehicle index; arrow keys, Enter and Escape; loading, empty and error states; browse fallback |
| Vehicle overview | `pages/vehicles/[make]/[model]/[year].astro` | wide layout, source date, linked leading complaints, official recalls and statistical flags; VIN lookup and evidence anchors |
| System explorer | `components/VehicleSystems.astro` | schematic SVG shortcuts and text buttons use existing category mappings; selected systems show their problems; all evidence remains available without JavaScript |
| Stepper | `components/Steps.astro` | 4 steps, across from 900px, down below |
| About-this-data callout | `components/AboutData.astro` | the AI-written / not-reviewed note, visible, not fine print |
| Icons | `components/Icon.astro` | inline SVG, 24px grid, 2px strokes, `currentColor` |

Vehicle search logic is in `src/lib/vehicle-search.ts`. Run `npm test` from `site/` for query,
index, ordering and empty-state tests (Node 22.6+ with type stripping). Run `npm run build`
to validate the exported vehicle pages against the content schemas. Vehicle JSON is generated
locally by `make export` and is not committed.
