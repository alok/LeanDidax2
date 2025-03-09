/-
  Examples of using the StateT monad transformer from LeanDidax2.StateTransformer.
  This file demonstrates how to use stateful computations in differentiable programs.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.StateTransformer
import LeanDidax2.Utils

namespace LeanDidax2.StateTransformerExamples

open LeanDidax2
open LeanDidax2.Monadic
open LeanDidax2.StateTransformer
open LeanDidax2.Utils

/-- Counter state for tracking function calls -/
structure Counter where
  count : Nat := 0
  lastValue : Float := 0.0

/-- Simple polynomial function with counter state -/
def polynomialWithState (x : Float) : DiffStateT Counter (Value Float) := do
  -- Get the current state
  let state ← get

  -- Increment the counter
  set { state with
        count := state.count + 1,
        lastValue := x }

  -- Compute the polynomial value
  let xv := seed x
  let two := constValue 2.0
  let one := constValue 1.0
  let result := xv * xv + two * xv + one

  pure result

/-- Gradient computation with state tracking -/
def gradWithState (x : Float) : DiffStateT Counter Float := do
  let result ← polynomialWithState x
  pure result.tangent

/--
  Example using a counter to track function calls
-/
def counterExample : IO Unit := do
  let initialCounter : Counter := {}

  -- Run computation with state
  runDiffStateIO initialCounter (polynomialWithState 3.0) (fun result finalState => do
    IO.println "StateT Counter Example"
    IO.println "---------------------"
    IO.println s!"Polynomial at x=3.0: {result}"
    IO.println s!"Function was called {finalState.count} time(s)"
    IO.println s!"Last input value: {finalState.lastValue}"
  )

  -- Run gradient computation with state
  runDiffStateIO initialCounter (gradWithState 3.0) (fun gradient gradState => do
    IO.println "\nGradient with state:"
    IO.println s!"Gradient at x=3.0: {gradient}"
    IO.println s!"Function was called {gradState.count} time(s)"
  )

/-- A simple stateful computation that accumulates the sum of derivatives -/
def accumulateDerivatives (xs : Array Float) : DiffStateT Float (Array Float) := do
  let mut results : Array Float := #[]

  for x in xs do
    let xv := seed x
    let two := constValue 2.0
    let one := constValue 1.0
    let f := xv * xv + two * xv + one

    -- Accumulate the derivative in the state
    let currentSum ← get
    let newSum := currentSum + f.tangent
    set newSum

    -- Add the original value to the results
    results := results.push f.primal

  pure results

/--
  Example using state to accumulate derivatives
-/
def accumulateExample : IO Unit := do
  let inputs := #[1.0, 2.0, 3.0, 4.0, 5.0]
  let initialSum := 0.0

  runDiffStateIO initialSum (accumulateDerivatives inputs) (fun values totalDerivative => do
    IO.println "Accumulated Derivatives Example"
    IO.println "------------------------------"
    IO.println s!"Input values: {inputs}"
    IO.println s!"Output values: {values}"
    IO.println s!"Sum of all derivatives: {totalDerivative}"

    -- Verify the result
    let expectedDerivatives := inputs.map (fun x => 2.0 * x + 2.0)
    let expectedSum := expectedDerivatives.foldl (· + ·) 0.0

    IO.println s!"Expected sum: {expectedSum}"
    IO.println s!"Correct?: {if Float.abs (totalDerivative - expectedSum) < 1e-10 then "Yes" else "No"}"
  )

/-- Use the ADState for detailed tracking of AD operations -/
def polynomialWithADState (x : Float) : DiffStateT ADState (Value Float) := do
  -- Enable tracing
  setTracing true

  -- Track each operation
  let xv := seed x
  trackOp "seed"
  trackTangent xv

  let two := constValue 2.0
  let one := constValue 1.0

  let x_squared ← trackedOp "multiply" xv xv (fun a b => a * b)
  let two_x ← trackedOp "multiply" two xv (fun a b => a * b)
  let sum1 ← trackedOp "add" x_squared two_x (fun a b => a + b)
  let result ← trackedOp "add" sum1 one (fun a b => a + b)

  pure result

/--
  Example using detailed ADState tracking
-/
def adStateExample : IO Unit := do
  let initialState : ADState := {}

  runDiffStateIO initialState (polynomialWithADState 3.0) (fun result finalState => do
    IO.println "ADState Tracking Example"
    IO.println "-----------------------"
    IO.println s!"Polynomial at x=3.0: {result}"
    IO.println s!"Operation count: {finalState.opCount}"
    IO.println s!"Maximum tangent value: {finalState.maxTangent}"

    if finalState.traceEnabled && finalState.opHistory.size > 0 then
      IO.println "\nOperation history:"
      for op in finalState.opHistory do
        IO.println s!" - {op}"

    -- Get operation summary
    runDiffStateIO finalState getOpSummary (fun summary _ => do
      IO.println s!"\nSummary: {summary}"
    )
  )

/--
  Run all state transformer examples
-/
def main : IO Unit := do
  counterExample
  IO.println "\n"
  accumulateExample
  IO.println "\n"
  adStateExample

end LeanDidax2.StateTransformerExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.StateTransformerExamples.main
