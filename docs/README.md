# Project Website (`docs/`)

This folder is a static GitHub Pages site for the public-facing project/paper page.

## Files

- `index.html`: content and section structure
- `styles.css`: visual style and responsive layout
- `script.js`: active nav, citation copy button, media placeholder handling
- `assets/figures/`: place figure images here
- `assets/replays/`: place replay clips or GIF thumbnails here
- `assets/icons/`: optional icons or logos

## Quick edits

1. Update title/authors/affiliation/abstract text in `index.html` near the Hero section comments.
2. Replace placeholder links in `index.html`:
   - `PAPER_LINK`
   - `ARXIV_LINK`
   - `REPO_LINK`
   - `README_LINK`
3. Replace figure placeholders:
   - `FIGURE_1_PATH`, `FIGURE_2_PATH`, ...
   - Put files in `assets/figures/` and update the corresponding `src`.
4. Replace replay placeholders:
   - `REPLAY_1_PATH`, `REPLAY_2_PATH`, ...
   - Put media in `assets/replays/` and update the `<source src="...">`.
5. Replace `BIBTEX_PLACEHOLDER` in the Citation section.

## GitHub Pages

In repository settings:

1. Open `Settings` -> `Pages`
2. Under "Build and deployment", set:
   - Source: `Deploy from a branch`
   - Branch: `main` (or your default branch)
   - Folder: `/docs`
3. Save and wait for deployment.
