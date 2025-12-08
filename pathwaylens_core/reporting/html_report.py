"""
HTML Report Generator for PathwayLens.
"""

import json
import os
from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime

class HTMLReportGenerator:
    """Generates interactive HTML reports for analysis results."""
    
    def __init__(self):
        self.logger = logger.bind(module="html_report")
        
    def generate_report(
        self,
        analysis_result: Dict[str, Any],
        output_file: str,
        plots: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Generate HTML report.
        
        Args:
            analysis_result: Analysis result dictionary
            output_file: Output file path
            plots: Dictionary of plot names to HTML strings (e.g. plotly div)
            
        Returns:
            True if successful
        """
        try:
            # Prepare data
            summary = analysis_result.get('summary', {})
            parameters = analysis_result.get('parameters', {})
            pathways = []
            
            # Extract pathways from result
            if 'database_results' in analysis_result:
                for db_name, db_res in analysis_result['database_results'].items():
                    for p in db_res.get('pathways', []):
                        # Flatten pathway info
                        p_flat = p.copy()
                        p_flat['database'] = db_name
                        p_flat['overlapping_genes'] = ", ".join(p.get('overlapping_genes', []))
                        pathways.append(p_flat)
            
            # Create HTML content
            html_content = self._create_html_content(summary, parameters, pathways, plots)
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
                
            self.logger.info(f"HTML report generated at {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return False
            
    def _create_html_content(
        self,
        summary: Dict[str, Any],
        parameters: Dict[str, Any],
        pathways: List[Dict[str, Any]],
        plots: Optional[Dict[str, str]]
    ) -> str:
        """Create the HTML string."""
        
        # Convert pathways to JSON for DataTables
        pathways_json = json.dumps(pathways)
        
        # Plots HTML
        plots_html = ""
        if plots:
            for name, plot_div in plots.items():
                plots_html += f"""
                <div class="card mb-4">
                    <div class="card-header">{name}</div>
                    <div class="card-body">
                        {plot_div}
                    </div>
                </div>
                """
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PathwayLens Analysis Report</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- DataTables CSS -->
    <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    
    <style>
        body {{ padding-top: 20px; background-color: #f8f9fa; }}
        .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header-section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="header-section">
            <h1 class="display-5">PathwayLens Analysis Report</h1>
            <p class="lead">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <hr>
            <div class="row">
                <div class="col-md-6">
                    <h5>Summary Statistics</h5>
                    <ul class="list-group">
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Total Pathways
                            <span class="badge bg-primary rounded-pill">{len(pathways)}</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Significant Pathways (FDR < 0.05)
                            <span class="badge bg-success rounded-pill">{len([p for p in pathways if p.get('adjusted_p_value', 1) < 0.05])}</span>
                        </li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h5>Parameters</h5>
                    <pre class="bg-light p-2 border rounded">{json.dumps(parameters, indent=2)}</pre>
                </div>
            </div>
        </div>

        <!-- Plots Section -->
        {plots_html}

        <!-- Pathways Table -->
        <div class="card">
            <div class="card-header">
                Pathway Results
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table id="pathwayTable" class="table table-striped table-hover" style="width:100%">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Name</th>
                                <th>Database</th>
                                <th>P-value</th>
                                <th>Adj. P-value</th>
                                <th>Size</th>
                                <th>Overlap</th>
                                <th>Genes</th>
                            </tr>
                        </thead>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <!-- DataTables JS -->
    <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
    <!-- Plotly JS -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <script>
        $(document).ready(function() {{
            var data = {pathways_json};
            
            $('#pathwayTable').DataTable({{
                data: data,
                columns: [
                    {{ data: 'pathway_id' }},
                    {{ data: 'pathway_name' }},
                    {{ data: 'database' }},
                    {{ 
                        data: 'p_value',
                        render: function(data) {{ return data.toExponential(3); }}
                    }},
                    {{ 
                        data: 'adjusted_p_value',
                        render: function(data) {{ return data.toExponential(3); }}
                    }},
                    {{ data: 'pathway_count' }},
                    {{ data: 'overlap_count' }},
                    {{ 
                        data: 'overlapping_genes',
                        render: function(data) {{
                            return '<span title="' + data + '">' + 
                                   (data.length > 50 ? data.substr(0, 50) + '...' : data) + 
                                   '</span>';
                        }}
                    }}
                ],
                order: [[4, 'asc']], // Sort by Adj. P-value
                dom: 'Bfrtip',
                pageLength: 25
            }});
        }});
    </script>
</body>
</html>
"""
