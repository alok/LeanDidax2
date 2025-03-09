/-
  Examples of using the LeanDidax batching functionality.
  This file provides concrete examples of using vmap and evaluateRange.
-/

import LeanDidax2.Basic
import LeanDidax2.Batch

namespace LeanDidax2.BatchExamples

open LeanDidax2
open LeanDidax2.Batch

/--
  Demonstrate basic batching functionality with a simple function
-/
def batchingExample : IO Unit := do
  -- Define a simple function to run batched
  let f (x : Value Float) : Value Float := x * x + 2 * x + 1

  -- Create a range of inputs
  let inputs : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0]

  -- Apply the function to all inputs and get the values
  let outputs := batchApply f inputs

  -- Apply the function to all inputs and get both values and derivatives
  let gradientsResults := batchGradient f inputs

  -- Print results
  IO.println "Batch processing example:"
  IO.println "------------------------"
  IO.println "f(x) = x² + 2x + 1"
  IO.println ""

  IO.println "Values only:"
  for i in [:inputs.size] do
    IO.println s!"f({inputs[i]!}) = {outputs[i]!}"

  IO.println "\nValues and derivatives:"
  for i in [:inputs.size] do
    let (value, derivative) := gradientsResults[i]!
    IO.println s!"f({inputs[i]!}) = {value}, f'({inputs[i]!}) = {derivative}"

  -- Use vmap to get an array of Value objects
  let valueResults := vmap f inputs

  IO.println "\nVmap results (as Value objects):"
  for i in [:inputs.size] do
    IO.println s!"f({inputs[i]!}) = {valueResults[i]!.primal}, f'({inputs[i]!}) = {valueResults[i]!.tangent}"

/--
  Demonstrate plotting a function and its derivative over a range
-/
def plotFunctionExample : IO Unit := do
  -- Define a more complex function
  let f (x : Value Float) : Value Float := x * x * x - 2 * x * x + x

  -- Evaluate the function over a range
  let results := evaluateRange f (-2.0) 2.0 20

  -- Print results in a format that could be plotted
  IO.println "Plot data for f(x) = x³ - 2x² + x"
  IO.println "--------------------------------"
  IO.println "x,f(x),f'(x)"

  for (x, y, dy) in results do
    IO.println s!"{x},{y},{dy}"

  -- Analyze the critical points (where derivative is zero)
  IO.println "\nCritical points (approximate):"
  for i in [0:results.size-1] do
    let (x1, _, dy1) := results[i]!
    let (x2, _, dy2) := results[i+1]!
    if dy1 * dy2 ≤ 0 then  -- Sign change indicates zero crossing
      let estimated_x := (x1 + x2) / 2.0
      IO.println s!"Critical point near x = {estimated_x}"

/--
  Run all batch examples
-/
def main : IO Unit := do
  batchingExample
  IO.println "\n"
  plotFunctionExample

end LeanDidax2.BatchExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.BatchExamples.main
