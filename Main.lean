import LeanDidax2

def main : IO Unit := do
  IO.println "LeanDidax2: Automatic Differentiation in Lean 4"
  IO.println "---------------------------------------------"
  IO.println "Run individual examples with:"
  IO.println "  lake env lean --run LeanDidax2/Examples.lean"
  IO.println "  lake env lean --run LeanDidax2/BatchExamples.lean"
  IO.println "  lake env lean --run LeanDidax2/CustomRulesExamples.lean"
  IO.println "  lake env lean --run LeanDidax2/ControlFlowExamples.lean"
  IO.println ""
  IO.println "Or use the Justfile commands:"
  IO.println "  just run-examples"
  IO.println "  just run-basic-examples"
  IO.println ""
  IO.println "For more information, see the README.md"
