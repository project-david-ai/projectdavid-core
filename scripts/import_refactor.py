import ast
import os
from pathlib import Path


class ImportRefactor(ast.NodeTransformer):
    """
    An AST transformer that rewrites specific import patterns.
    It changes:
      - `from entities_api...` to `from src.api.entities_api...`
      - `from ..services...` to `from src.api.entities_api.services...`
    """

    def __init__(self, file_path, project_root):
        self.file_path = file_path
        self.project_root = project_root
        # Calculate the module path relative to the src directory
        relative_path = file_path.relative_to(project_root / "src")
        self.module_parts = list(relative_path.parts[:-1])

    def visit_ImportFrom(self, node):
        # Handle cases like `from entities_api.dependencies ...`
        if node.module and node.module.startswith("entities_api"):
            new_module_name = f"src.api.{node.module}"
            print(f"  Rewriting: from {node.module} -> from {new_module_name}")
            node.module = new_module_name
            return node

        # Handle relative imports like `from ..services ...`
        if node.level > 0:  # e.g., level=1 for `from .`, level=2 for `from ..`
            # Calculate the path after moving up `level` directories
            effective_path_parts = self.module_parts[
                : len(self.module_parts) - (node.level - 1)
            ]

            # Construct the new absolute path from src
            # Add the original module if it existed (e.g., the 'services' in '..services')
            new_module_parts = ["src"] + effective_path_parts
            if node.module:
                new_module_parts.append(node.module)

            new_module_name = ".".join(new_module_parts)

            # Only rewrite if it's a relative import within our package
            if "entities_api" in new_module_name:
                print(f"  Rewriting relative import: -> from {new_module_name}")
                node.module = new_module_name
                node.level = 0  # Set level to 0 for absolute import

        return node


def refactor_directory(directory):
    project_root = Path(__file__).parent.resolve()
    target_dir = project_root / directory

    print(f"--- Starting import refactoring in: {target_dir} ---")

    for file_path in target_dir.rglob("*.py"):
        print(f"\nProcessing file: {file_path.relative_to(project_root)}")

        try:
            with open(file_path, "r", encoding="utf-8") as source_file:
                original_code = source_file.read()

            tree = ast.parse(original_code)
            transformer = ImportRefactor(file_path, project_root)
            new_tree = transformer.visit(tree)

            # Ensure the tree is still valid
            ast.fix_missing_locations(new_tree)

            # Generate the new code
            new_code = ast.unparse(new_tree)

            if original_code != new_code:
                print(f"  File modified.")
                with open(file_path, "w", encoding="utf-8") as source_file:
                    source_file.write(new_code)
            else:
                print("  No changes needed.")

        except Exception as e:
            print(f"  ERROR processing {file_path}: {e}")

    print("\n--- Refactoring complete! ---")
    print("Please review the changes with 'git diff' before committing.")


if __name__ == "__main__":
    # Define the directory to refactor.
    # Based on your structure, this is the correct target.
    code_directory = "src/api/entities_api"
    refactor_directory(code_directory)
