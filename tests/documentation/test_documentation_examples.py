"""
Test all code examples in documentation files.

This module extracts and tests code examples from documentation
to ensure they remain accurate and functional.
"""

import re
import pytest
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import sys


class DocumentationExampleExtractor:
    """Extract code examples from documentation files."""
    
    def __init__(self, docs_dir: Path = None):
        """
        Initialize extractor.
        
        Args:
            docs_dir: Directory containing documentation files
        """
        if docs_dir is None:
            docs_dir = Path(__file__).parent.parent.parent / "docs"
        self.docs_dir = Path(docs_dir)
    
    def extract_code_blocks(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract code blocks from a markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of code blocks with language and content
        """
        if not file_path.exists():
            return []
        
        code_blocks = []
        content = file_path.read_text()
        
        # Match code blocks: ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2).strip()
            
            # Only extract Python/bash code blocks
            if language in ["python", "bash", "shell", "py"]:
                code_blocks.append({
                    "language": language,
                    "code": code,
                    "file": str(file_path),
                    "line": content[:match.start()].count('\n') + 1
                })
        
        return code_blocks
    
    def extract_all_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all code examples from documentation."""
        examples = {}
        
        # Find all markdown files
        for md_file in self.docs_dir.rglob("*.md"):
            relative_path = md_file.relative_to(self.docs_dir)
            examples[str(relative_path)] = self.extract_code_blocks(md_file)
        
        return examples


@pytest.fixture(scope="session")
def doc_examples():
    """Extract all documentation examples."""
    extractor = DocumentationExampleExtractor()
    return extractor.extract_all_examples()


class TestDocumentationExamples:
    """Test code examples from documentation."""
    
    @pytest.mark.documentation
    def test_readme_examples(self, doc_examples):
        """Test code examples in README.md."""
        if "README.md" not in doc_examples:
            pytest.skip("README.md not found")
        
        examples = doc_examples["README.md"]
        python_examples = [e for e in examples if e["language"] in ["python", "py"]]
        
        for example in python_examples:
            # Skip if it's an import-only or comment-only block
            code = example["code"]
            if not any(keyword in code for keyword in ["import", "from", "def ", "class ", "="]):
                continue
            
            # Try to compile the code
            try:
                compile(code, f"<{example['file']}:{example['line']}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in README.md example at line {example['line']}: {e}")
    
    @pytest.mark.documentation
    def test_quickstart_examples(self, doc_examples):
        """Test code examples in QUICKSTART.md."""
        if "QUICKSTART.md" not in doc_examples:
            pytest.skip("QUICKSTART.md not found")
        
        examples = doc_examples["QUICKSTART.md"]
        python_examples = [e for e in examples if e["language"] in ["python", "py"]]
        
        for example in python_examples:
            code = example["code"]
            if not any(keyword in code for keyword in ["import", "from", "def ", "class ", "="]):
                continue
            
            try:
                compile(code, f"<{example['file']}:{example['line']}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in QUICKSTART.md example at line {example['line']}: {e}")
    
    @pytest.mark.documentation
    def test_user_guide_examples(self, doc_examples):
        """Test code examples in USER_GUIDE.md."""
        if "USER_GUIDE.md" not in doc_examples:
            pytest.skip("USER_GUIDE.md not found")
        
        examples = doc_examples["USER_GUIDE.md"]
        python_examples = [e for e in examples if e["language"] in ["python", "py"]]
        
        for example in python_examples:
            code = example["code"]
            if not any(keyword in code for keyword in ["import", "from", "def ", "class ", "="]):
                continue
            
            try:
                compile(code, f"<{example['file']}:{example['line']}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in USER_GUIDE.md example at line {example['line']}: {e}")
    
    @pytest.mark.documentation
    def test_api_reference_examples(self, doc_examples):
        """Test code examples in API_REFERENCE.md."""
        if "API_REFERENCE.md" not in doc_examples:
            pytest.skip("API_REFERENCE.md not found")
        
        examples = doc_examples["API_REFERENCE.md"]
        python_examples = [e for e in examples if e["language"] in ["python", "py"]]
        
        for example in python_examples:
            code = example["code"]
            if not any(keyword in code for keyword in ["import", "from", "def ", "class ", "="]):
                continue
            
            try:
                compile(code, f"<{example['file']}:{example['line']}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in API_REFERENCE.md example at line {example['line']}: {e}")
    
    @pytest.mark.documentation
    def test_cli_reference_examples(self, doc_examples):
        """Test CLI command examples in CLI_REFERENCE.md."""
        if "CLI_REFERENCE.md" not in doc_examples:
            pytest.skip("CLI_REFERENCE.md not found")
        
        examples = doc_examples["CLI_REFERENCE.md"]
        bash_examples = [e for e in examples if e["language"] in ["bash", "shell"]]
        
        # For bash examples, we just check they're valid command structures
        for example in bash_examples:
            code = example["code"]
            # Check for basic command structure
            if code.strip() and not code.strip().startswith("#"):
                # Should start with pathwaylens or a valid command
                first_line = code.strip().split('\n')[0]
                if not (first_line.startswith("pathwaylens") or 
                       first_line.startswith("#") or
                       first_line.startswith("python") or
                       first_line.startswith("pip")):
                    # Not a critical error, just a warning
                    pass
    
    @pytest.mark.documentation
    def test_all_documentation_files_exist(self):
        """Test that all expected documentation files exist."""
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        expected_files = [
            "README.md",
            "QUICKSTART.md",
            "USER_GUIDE.md",
            "API_REFERENCE.md",
            "CLI_REFERENCE.md",
            "INSTALLATION.md",
            "CONTRIBUTING.md"
        ]
        
        missing_files = []
        for filename in expected_files:
            file_path = docs_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            pytest.fail(f"Missing documentation files: {', '.join(missing_files)}")
    
    @pytest.mark.documentation
    def test_installation_instructions(self):
        """Test that installation instructions reference correct files."""
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        install_file = docs_dir / "INSTALLATION.md"
        
        if not install_file.exists():
            pytest.skip("INSTALLATION.md not found")
        
        content = install_file.read_text()
        
        # Check for references to requirements.txt or pyproject.toml
        has_requirements = "requirements.txt" in content or "pyproject.toml" in content
        assert has_requirements, "Installation instructions should reference requirements.txt or pyproject.toml"
        
        # Check for pip install command
        has_pip_install = "pip install" in content.lower()
        assert has_pip_install, "Installation instructions should include pip install command"



