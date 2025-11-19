# PathwayLens Research Upgrade Roadmap

**Goal**: Transform PathwayLens into a highly intricate, robust, and publication-ready computational tool for multi-omics pathway analysis.

## 🎯 Core Objectives
1.  **Robustness**: Comprehensive testing, input validation, and error handling.
2.  **Reproducibility**: Full provenance tracking, version control of data/methods, and deterministic execution.
3.  **Advanced Statistics**: rigorous statistical methods (e.g., bootstrapping, permutation tests, confidence intervals, odds ratios).
4.  **Publication-Ready Visualization**: High-quality, customizable plots with export options (SVG, PDF).
5.  **Interoperability**: Support for standard formats (BioPAX, CX, Cytoscape) and easy integration with other tools.

---

## 🗓️ Phase 1: Foundation & Robustness (Immediate Priority)
Focus on fixing existing issues and establishing a solid base.

- [ ] **Fix Test Suite**
    - [ ] Update `ORAEngine` tests to match implementation (or update implementation).
    - [ ] Update `GSEAEngine` tests.
    - [ ] Resolve Pydantic V2 warnings.
    - [ ] Ensure 100% pass rate for core engines.
- [ ] **Enhance Input Validation**
    - [ ] Implement strict parameter validation in all engines.
    - [ ] Add data quality checks (missing values, duplicates, format consistency).
- [ ] **Modularize Core Logic**
    - [ ] Extract statistical calculations (odds ratio, CI) into reusable helper methods.
    - [ ] Refactor large methods into smaller, testable units.

## 📊 Phase 2: Advanced Statistical Methods
Implement rigorous statistical tools required for high-impact research.

- [ ] **Extended ORA Statistics**
    - [ ] Calculate Odds Ratio and Confidence Intervals for enrichment.
    - [ ] Implement alternative tests (Fisher's Exact Test, Chi-Square).
- [ ] **Robust GSEA**
    - [ ] Add bootstrapping for confidence intervals of Enrichment Scores.
    - [ ] Implement "Leading Edge" analysis improvements.
- [ ] **Multi-Testing Correction**
    - [ ] Add more correction methods (Storey's q-value).
    - [ ] Visualize p-value distributions.

## 🔍 Phase 3: Reproducibility & Provenance
Ensure every result can be traced back to its source.

- [ ] **Provenance Tracking**
    - [ ] Record exact versions of databases (KEGG, Reactome) used.
    - [ ] Log all parameters, timestamps, and software versions.
    - [ ] Generate a "Analysis Manifest" file for every run.
- [ ] **Deterministic Execution**
    - [ ] Ensure random seeds are handled correctly for permutations.
    - [ ] Verify consistency across runs.

## 📈 Phase 4: Visualization & Reporting
Create visuals that are ready for papers.

- [ ] **Enhanced Visualizations**
    - [ ] Interactive plots (using Plotly) with static export (SVG/PDF).
    - [ ] Volcano plots, Dot plots, Enrichment maps, Upset plots.
    - [ ] Network views of pathway overlaps.
- [ ] **Comprehensive Reports**
    - [ ] Generate HTML/PDF reports summarizing analysis.
    - [ ] Include methods section text generation (auto-drafting methods for papers).

## 🔗 Phase 5: Interoperability & Data Handling
Handle various data types and integrate with the ecosystem.

- [ ] **Data Connectors**
    - [ ] Direct fetch from GEO/SRA (if possible/allowed).
    - [ ] Support for more single-cell formats (Seurat conversion).
- [ ] **Export Formats**
    - [ ] Export results to Cytoscape (CX format).
    - [ ] Export to standard tabular formats with full metadata.

---

## 📝 Implementation Log

### [Date: 2024-12-19] - Initial Assessment
- Reviewed `UPGRADE_ROADMAP.md` and `TEST_RESULTS.md`.
- Identified test failures in ORA and GSEA engines.
- Planned immediate fixes for robustness (Phase 1).
