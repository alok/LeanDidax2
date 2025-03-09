/-
  Examples of using the LeanDidax control flow primitives for differentiable programming.
  This file demonstrates how to use conditionals within automatic differentiation.
-/

import LeanDidax2.Basic
import LeanDidax2.ControlFlow

namespace LeanDidax2.ControlFlowExamples

open LeanDidax2
open LeanDidax2.ControlFlow

/--
  A smooth approximation of relu(x) = max(0, x).
  This is implemented using a standard relu as reference.
-/
def reluExample : IO Unit := do
  -- Define a range of x values to test
  let xs : List Float := [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

  IO.println "ReLU Function Example"
  IO.println "-------------------"
  IO.println "f(x) = max(0, x) implemented with control flow"
  IO.println ""

  for x in xs do
    let input := seed x

    -- Get result using control flow
    let result := reluWithControlFlow input

    -- The expected derivative is 0 for x < 0, and 1 for x > 0
    -- At x = 0, the derivative is technically undefined, but we'll use 0
    let expectedDx := if x > 0.0 then 1.0 else 0.0

    IO.println s!"At x = {x}:"
    IO.println s!"  ReLU(x) = {result.primal}"
    IO.println s!"  d/dx ReLU(x) = {result.tangent}"
    IO.println s!"  Expected derivative = {expectedDx}"
    IO.println s!"  Correct? {if result.tangent == expectedDx then "Yes" else "No"}"
    IO.println ""

/--
  A piecewise function using the cond primitive, demonstrating
  how derivatives flow through conditional branches.

  f(x) = { x^2     if x > 0
         { -x      otherwise
-/
def condExample : IO Unit := do
  -- Define a range of x values to test
  let xs : List Float := [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

  IO.println "Conditional Function Example"
  IO.println "--------------------------"
  IO.println "f(x) = x^2 if x > 0, -x otherwise"
  IO.println ""

  for x in xs do
    let input := seed x

    -- Get result using cond
    let result := conditionalFunction input

    -- Compute the expected result and derivative manually
    let expectedValue := if x > 0.0 then x * x else -x
    let expectedDx := if x > 0.0 then 2.0 * x else -1.0

    IO.println s!"At x = {x}:"
    IO.println s!"  f(x) = {result.primal}"
    IO.println s!"  Expected value = {expectedValue}"
    IO.println s!"  d/dx f(x) = {result.tangent}"
    IO.println s!"  Expected derivative = {expectedDx}"
    IO.println s!"  Value correct? {if Float.abs (result.primal - expectedValue) < 1e-10 then "Yes" else "No"}"
    IO.println s!"  Derivative correct? {if Float.abs (result.tangent - expectedDx) < 1e-10 then "Yes" else "No"}"
    IO.println ""

/--
  Example demonstrating the switch primitive, which selects one
  of several functions based on an index.
-/
def switchTest : IO Unit := do
  -- Define x values to test
  let xs : List Float := [1.0, 2.0]

  -- The functions in our switch
  let functionNames : List String := ["x^2", "x^3", "x+5", "sin(x)", "e^x"]

  -- For each function, define the expected value and derivative at each point
  let expectedResults : List (List (Float × Float)) := [
    -- For x = 1.0
    [(1.0, 2.0),     -- x^2: value = 1, derivative = 2
     (1.0, 3.0),     -- x^3: value = 1, derivative = 3
     (6.0, 1.0),     -- x+5: value = 6, derivative = 1
     (0.84147, 0.5403),  -- sin(x): approximate values
     (2.7183, 2.7183)],  -- e^x: approximate values

    -- For x = 2.0
    [(4.0, 4.0),     -- x^2: value = 4, derivative = 4
     (8.0, 12.0),    -- x^3: value = 8, derivative = 12
     (7.0, 1.0),     -- x+5: value = 7, derivative = 1
     (0.9093, -0.4161),  -- sin(x): approximate values
     (7.3891, 7.3891)]   -- e^x: approximate values
  ]

  IO.println "Switch Function Example"
  IO.println "----------------------"
  IO.println "Selecting between different functions based on an index"
  IO.println ""

  -- Test each x value
  for i in [0:xs.length] do
    let x := xs[i]!
    let input := seed x

    IO.println s!"For x = {x}:"

    -- Test each branch
    for j in [0:functionNames.length] do
      let funcName := functionNames[j]!
      if let some xResults := expectedResults[i]? then
        if let some expectedPair := xResults[j]? then
          let (expectedValue, expectedDx) := expectedPair

          -- Apply the switch
          let result := ControlFlow.switchExample input j

          IO.println s!"  Branch {j} ({funcName}):"
          IO.println s!"    Value = {result.primal}"
          IO.println s!"    Expected = {expectedValue}"
          IO.println s!"    Derivative = {result.tangent}"
          IO.println s!"    Expected dx = {expectedDx}"
          IO.println s!"    Correct? {if Float.abs (result.primal - expectedValue) < 0.01 &&
                                Float.abs (result.tangent - expectedDx) < 0.01 then "Yes" else "No"}"

    IO.println ""

/--
  Run all control flow examples.
-/
def runAllExamples : IO Unit := do
  reluExample
  IO.println "--------------------------------\n"
  condExample
  IO.println "--------------------------------\n"
  switchTest

end LeanDidax2.ControlFlowExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.ControlFlowExamples.runAllExamples
