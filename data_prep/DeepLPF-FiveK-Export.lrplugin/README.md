# DeepLPF FiveK Export — Lightroom Classic plug-in

Renders the FiveK **Input** and **Expert-C** collections to
`~/fivek_export/input` and `~/fivek_export/output` with the exact settings
DeepLPF expects — **PNG, sRGB, 8-bit, long edge 512, no enlarge** — preserving
the original `aXXXX-...` filenames. It automates the two manual Export dialogs so
the settings can't be got wrong.

## Install (one time)

1. Open **Adobe Lightroom Classic**.
2. Open the FiveK catalogue: **File ▸ Open Catalog…** →
   `~/fivek/fivek_dataset/raw_photos/fivek.lrcat` (let the upgrade finish).
3. **File ▸ Plug-in Manager… ▸ Add**, then select this folder:
   `…/deeplpf-image-enhancement/data_prep/DeepLPF-FiveK-Export.lrplugin`
   → **Add Plug-in** → **Done**. You should see "DeepLPF FiveK Export" enabled.

## Run

1. **File ▸ Plug-in Extras ▸ Export FiveK for DeepLPF…**
   (also under **Library ▸ Plug-in Extras** in the Library module).
2. In the dialog, two dropdowns list every collection in the catalogue. Pick:
   - **Input** → *InputAsShotZeroed* (this is the correct rendering; it is the
     one the bundled reference inputs came from. Do **not** use *Input
     with Daylight WhiteBalance minus 1.5* — it is −1.5 EV too dark and
     fails the verify step.)
   - **Expert-C target** → `Experts / C`
   The plug-in tries to preselect these, but confirm them.
3. Click **Export**. It exports the Input collection to `~/fivek_export/input`,
   then the target to `~/fivek_export/output`, showing Lightroom's progress bar.
   ~5000 raws render each way, so it takes a while.
4. When it reports "Export complete", tell the terminal session **"exported"** and
   the `organise_fivek.py` + `verify_dataset.py` steps run automatically.

## Notes

- **Overwrite:** re-running overwrites existing files, so it's safe to redo (e.g.
  if `verify_dataset.py` reports a large `mean|Δ|` and you want a different Input
  rendering — just re-run and pick another).
- **Filenames:** original names are kept, so the `aXXXX-` id prefix the data
  loader pairs on is preserved.
- **Dialog is slow to appear:** it counts the photos in every collection first,
  which on the FiveK catalogue means walking tens of thousands of photo records.
  Give it a moment.
- **If it errors:** a dialog shows the message — copy it into the terminal session
  and it'll be fixed. (The plug-in couldn't be run-tested outside Lightroom, so a
  first-run tweak to the SDK settings is possible.)
