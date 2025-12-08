# Test Suite Notes

## Known Issues

### Database Availability
The tests currently show warnings: "Database kegg is not available"

This is **expected behavior** for the test suite because:
1. Pathway databases (KEGG, Reactome, GO, etc.) require downloading large datasets
2. These databases are not included in the repository due to size and licensing
3. The tool gracefully handles missing databases and continues processing

**For production use**, databases should be set up using:
```bash
# This functionality will be implemented in future versions
pathwaylens database download kegg reactome go
```

### Current Test Behavior
- Tests run successfully and generate output files
- HTML visualizations are created
- JSON summaries are generated  
- The tool processes gene lists correctly
- Pathway enrichment returns empty results (expected without databases)

### What's Being Tested
Even without pathway databases, the tests verify:
- ✅ CLI argument parsing
- ✅ Input file format detection (DESeq2, MaxQuant, generic, etc.)
- ✅ Gene list extraction
- ✅ Data type handling (bulk, singlecell, spatial, etc.)
- ✅ Omic type processing (transcriptomics, proteomics, etc.)
- ✅ Output file generation
- ✅ Visualization creation
- ✅ Error handling

### For Full Testing
To test with actual pathway enrichment results, you would need to:
1. Set up pathway databases (future feature)
2. Or use mock pathway data for unit tests

The current test suite validates the **CLI interface and data processing pipeline**, which is the primary goal for release readiness.
