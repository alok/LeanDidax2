/-
  Examples of using the ExceptT adapter for error handling in differentiable computations.
  This file demonstrates how to handle errors like domain errors, division by zero, etc.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.ExceptTransformer
import LeanDidax2.StateTransformer
import LeanDidax2.Utils

namespace LeanDidax2.ExceptTransformerExamples

open LeanDidax2
open LeanDidax2.Monadic
open LeanDidax2.ExceptTransformer
open LeanDidax2.StateTransformer
open LeanDidax2.Utils

/-- Example of a function that can encounter domain errors -/
def riskyFunction (x : Value Float) : DiffExceptT (Value Float) := do
  -- First do a safe operation
  let square := x * x

  -- Then a risky one: logarithm of a number that could be negative
  let result ← safelog (square - constValue 4.0)

  -- Some more computation with the result
  pure (result * constValue 2.0)

/-- Example of error handling with a fallback value -/
def handleErrors (x : Float) : IO Unit := do
  let computation := riskyFunction (seed x)

  runDiffExceptIO
    computation
    (fun result => do
      IO.println s!"Success! Result = {result}")
    (fun error => do
      IO.println s!"Failed: {error}")

/-- Example of using error handling with a default value -/
def withDefaultValue (x : Float) : IO Unit := do
  let computation := riskyFunction (seed x)
  let withDefault := tryWithDefault computation { primal := 0, tangent := 0 }

  -- Run and extract the result
  let result := runDiff withDefault

  IO.println s!"Result with default = {result}"
  IO.println s!"(If x is between -2 and 2, this would normally be an error)"

/-- Example of catching specific errors -/
def catchSpecificError (x : Float) : IO Unit := do
  let xv := seed x

  -- Define a computation that will fail for x ≤ 0
  let riskyComp := safelog xv

  -- Define an error handler that substitutes a default value
  let handled := catchDomain riskyComp (fun msg =>
    pure { primal := 0, tangent := 0 }
  )

  runDiffExceptIO
    handled
    (fun result => do
      IO.println s!"Final result = {result}")
    (fun error => do
      IO.println s!"Other error: {error}")

/-- SafeCalculator state for tracking operations and errors -/
structure SafeCalculator where
  result : Float := 0.0
  errorCount : Nat := 0
  lastError : Option String := none
  operationCount : Nat := 0

/-- Stateful computation that tracks errors -/
def safeCalculation (xs : Array Float) : DiffExceptStateT SafeCalculator (Array Float) := do
  let mut results : Array Float := #[]

  for x in xs do
    let xv := seed x

    -- Track the operation
    modify (fun s => { s with operationCount := s.operationCount + 1 })

    -- Try a risky division operation
    let divResult ← do
      try
        let res ← safediv (constValue 1.0) xv
        pure (some res)
      catch e =>
        -- Track the error
        modify (fun s => {
          s with
          errorCount := s.errorCount + 1,
          lastError := some (toString e)
        })
        pure none

    -- Process the result
    match divResult with
    | some res =>
      results := results.push res.primal
    | none =>
      -- Use a default value
      results := results.push 0.0

  pure results

/-- Example of combining state tracking with error handling -/
def stateAndErrorExample : IO Unit := do
  let inputs := #[2.0, 1.0, 0.0, -1.0, 0.5]
  let initialState : SafeCalculator := {}

  runDiffExceptStateIO
    initialState
    (safeCalculation inputs)
    (fun results state => do
      IO.println "State and Error Handling Example"
      IO.println "-------------------------------"
      IO.println s!"Input values: {inputs}"
      IO.println s!"Results: {results}"
      IO.println s!"Operations performed: {state.operationCount}"
      IO.println s!"Errors encountered: {state.errorCount}"
      if let some err := state.lastError then
        IO.println s!"Last error: {err}"
    )
    (fun error => do
      IO.println s!"Computation failed: {error}"
    )

/-- Example of validating computation results -/
def validationExample (x : Float) : IO Unit := do
  -- Define a computation
  let comp : DiffExceptT (Value Float) := do
    let xv := seed x
    let result := xv * xv - constValue 10.0

    -- Validate that the result is positive
    if result.primal < 0 then
      throw (DiffError.domainError s!"Result is negative: {result.primal}")

    pure result

  runDiffExceptIO
    comp
    (fun result => IO.println s!"Valid result: {result}")
    (fun error => IO.println s!"Validation failed: {error}")

/--
  A potentially unstable computation that must be validated.
  This function computes 1/(x^2 - 4), which has singularities at x=±2.
-/
def unstableFraction (x : Float) : DiffExceptT (Value Float) := do
  let xv := seed x
  let x_squared := xv * xv
  let denominator := x_squared - constValue 4.0

  -- Check for near-zero denominator to avoid numerical instability
  if Float.abs denominator.primal < 1e-6 then
    throw (DiffError.numericalError s!"Near-singularity at x ≈ {x}")

  safediv (constValue 1.0) denominator

/-- Visualize a function's valid domain by checking a range of values -/
def domainVisualization : IO Unit := do
  IO.println "Domain Visualization Example"
  IO.println "-------------------------"
  IO.println "Function f(x) = 1/(x^2 - 4)"
  IO.println "Checking x values from -3 to 3:"

  let min := -3.0
  let max := 3.0
  let steps := 12
  let stepSize := (max - min) / steps.toFloat

  for i in [:steps + 1] do
    let x := min + i.toFloat * stepSize

    runDiffExceptIO
      (unstableFraction x)
      (fun result => IO.println s!"x = {x}, f(x) = {result.primal}")
      (fun error => IO.println s!"x = {x}, Error: {error}")

/--
  Run all except transformer examples
-/
def main : IO Unit := do
  -- Basic error handling examples
  IO.println "\n=== Basic Error Handling ==="
  handleErrors 3.0  -- Should succeed
  handleErrors 1.0  -- Should fail (domain error in log)

  IO.println "\n=== Using Default Values ==="
  withDefaultValue 3.0
  withDefaultValue 1.0

  IO.println "\n=== Catching Specific Errors ==="
  catchSpecificError 2.0   -- Will succeed
  catchSpecificError (-1.0) -- Will catch domain error

  IO.println "\n=== Stateful Error Handling ==="
  stateAndErrorExample

  IO.println "\n=== Validation Example ==="
  validationExample 4.0  -- Valid (result is positive)
  validationExample 2.0  -- Invalid (result is negative)

  IO.println "\n=== Domain Visualization ==="
  domainVisualization

end LeanDidax2.ExceptTransformerExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.ExceptTransformerExamples.main
