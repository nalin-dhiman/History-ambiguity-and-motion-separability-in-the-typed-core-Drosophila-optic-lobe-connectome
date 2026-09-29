# External numerical inputs

- Male optic-lobe connectivity: FlyEM optic-lobe v1.1 neuPrint tables, associated with Nern et al., DOI [10.1038/s41586-025-08746-0](https://doi.org/10.1038/s41586-025-08746-0). The published matching workbook comes from the article's supplementary data. Source release: https://www.janelia.org/project-team/flyem/optic-lobe . The source identifies these data as CC BY.
- Female connectivity: FlyWire FAFB v783 proofread connections and root IDs. FlyWire public data use CC BY-NC 4.0; see https://home.flywire.ai/guidelines . Source publications include Zheng et al. (2018), Dorkenwald et al. (2022, 2024), Schlegel et al. (2024) and Matsliah et al. (2024).
- Female visual annotations: https://github.com/murthylab/visual-system-parts-list , pinned commit and individual file checksums in `cross_specimen/annotation_sources.json`. The repository license notice is downloaded alongside the metadata; it does not override the underlying reconstruction data terms.
- Female column assignments: the checksum-pinned Codex v783 snapshot listed in `cross_specimen/annotation_sources.json`. A changed snapshot is rejected by the downloader.
- Optional model arrays: exported numerical FlyVis connectivity, associated with Lappalainen et al., DOI [10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3). Upstream source: https://github.com/TuragaLab/flyvis . Preserve its MIT license notice when redistributing exported model data.

These resources are external inputs. This code update does not bundle raw reconstructions, new numerical outcomes or model checkpoints, and does not relicense third-party data or dependencies. Snapshot URLs and checksums support exact input identification.
