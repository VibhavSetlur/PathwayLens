from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import json
from loguru import logger
import zipfile
import nbformat as nbf


def generate_report(result: Any, output_dir: str, format: str = "html") -> Dict[str, str]:
    """
    Generate a minimal report artifact for a workflow result.
    This is a placeholder to enable wiring and can be expanded.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if format == "json":
            out_path = out_dir / "report.json"
            # Attempt to serialize pydantic models
            if hasattr(result, "model_dump_json"):
                out_path.write_text(result.model_dump_json(indent=2))
            else:
                out_path.write_text(json.dumps(result, default=str, indent=2))
            return {"json": str(out_path)}
        elif format == "notebook":
            nb = nbf.v4.new_notebook()
            cells = []
            cells.append(nbf.v4.new_markdown_cell("# PathwayLens Report"))
            cells.append(nbf.v4.new_markdown_cell("This notebook reproduces core outputs."))
            payload = result.model_dump_json(indent=2) if hasattr(result, "model_dump_json") else json.dumps(result, default=str, indent=2)
            cells.append(nbf.v4.new_code_cell("result = " + repr(payload)))
            nb["cells"] = cells
            nb_path = out_dir / "report.ipynb"
            with nb_path.open("w") as f:
                nbf.write(nb, f)
            return {"notebook": str(nb_path)}
        elif format == "zip":
            # create both html and json then zip
            artifacts = generate_report(result, output_dir, format="html")
            artifacts.update(generate_report(result, output_dir, format="json"))
            zip_path = out_dir / "report_artifacts.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for _, fpath in artifacts.items():
                    p = Path(fpath)
                    if p.exists():
                        zf.write(p, arcname=p.name)
            artifacts["zip"] = str(zip_path)
            return artifacts
        else:
            # Basic HTML wrapper
            out_path = out_dir / "report.html"
            content = "<html><body><pre>" + (
                result.model_dump_json(indent=2) if hasattr(result, "model_dump_json") else json.dumps(result, default=str, indent=2)
            ) + "</pre></body></html>"
            out_path.write_text(content)
            return {"html": str(out_path)}
    except Exception as exc:
        logger.error(f"Failed to generate report: {exc}")
        return {}


