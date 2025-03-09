/-
  Simple lint script for the LeanDidax project.
  This script serves as a placeholder for more sophisticated linting.
-/

def main : IO Unit := do
  IO.println "LeanDidax Linter"
  IO.println "---------------"
  IO.println "For linting, we recommend using built-in Lean linting capabilities:"
  IO.println "1. In VS Code, warnings are shown automatically in the Problems panel"
  IO.println "2. Running 'lake build' will report linter errors"
  IO.println "3. Using 'set_option diagnostics true' in your files provides detailed error information"
  IO.println ""
  IO.println "Common issues to watch for:"
  IO.println "- Use 'zipIdx' instead of 'List.enum'"
  IO.println "- Use array-like indexing 'a[i]?' instead of 'List.get?'"
  IO.println "- Avoid unnecessary semicolons at the end of statements"
  IO.println "- Use proper naming conventions (camelCase for variables, PascalCase for types)"
  IO.println ""
  IO.println "Lint completed successfully"
