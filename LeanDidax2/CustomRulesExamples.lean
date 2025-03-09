/-
  Examples of using custom derivative rules in LeanDidax.
-/

import LeanDidax2.Basic
import LeanDidax2.CustomRules

namespace LeanDidax2.CustomRulesExamples

open LeanDidax2
open LeanDidax2.CustomRules

/--
  Example of using a custom derivative rule for a complex function.

  This demonstrates how to define a function with a custom derivative
  rather than letting the autodiff system derive it automatically.
-/
def customDerivativeExample : IO Unit := do
  -- Define a function with a custom derivative rule
  -- f(x) = x^3 sin(x), which has derivative f'(x) = 3x^2 sin(x) + x^3 cos(x)
  let f (x : Float) : Float := x * x * x * Float.sin x
  let df (x : Float) : Float := 3 * x * x * Float.sin x + x * x * x * Float.cos x

  -- Define test points
  let xs : List Float := [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

  IO.println "Custom Derivative Example"
  IO.println "-----------------------"
  IO.println "f(x) = x^3 sin(x)"
  IO.println "f'(x) = 3x^2 sin(x) + x^3 cos(x)"
  IO.println ""

  -- Compare the custom derivative with numerical approximation
  for x in xs do
    -- Custom derivative
    let customValue := defCustomFn "cubic_sine" f df (seed x)

    -- Numerically approximate the derivative for comparison
    let h := 1e-5
    let numApprox := (f (x + h) - f x) / h

    IO.println s!"At x = {x}:"
    IO.println s!"  Function value: {customValue.primal}"
    IO.println s!"  Custom derivative: {customValue.tangent}"
    IO.println s!"  Numerical approximation: {numApprox}"
    IO.println s!"  Difference: {Float.abs (customValue.tangent - numApprox)}"
    IO.println ""

/--
  Example of using the rule registry to apply custom derivatives.
-/
def registryExample : IO Unit := do
  -- Create test data
  let xs : List Float := [0.5, 1.0, 1.5, 2.0]

  -- Get the predefined registry
  let registry := floatRegistry

  IO.println "Rule Registry Example"
  IO.println "-------------------"
  IO.println "Applying different rules from the registry:"
  IO.println ""

  -- Test each rule in the registry
  let ruleNames := ["square", "exp", "log", "sin", "cos"]

  for name in ruleNames do
    IO.println s!"Rule: {name}"
    for x in xs do
      let result := applyNamedRule registry name (seed x)
      IO.println s!"  f({x}) = {result.primal}, f'({x}) = {result.tangent}"
    IO.println ""

/--
  Define a complex function using custom rules to improve efficiency.
-/
def compositeFunction (x : Value Float) : Value Float :=
  -- This function is: f(x) = sin(x^2) * exp(cos(x))
  let registry := floatRegistry

  let xSquared := applyNamedRule registry "square" x
  let sinXSquared := applyNamedRule registry "sin" xSquared

  let cosX := applyNamedRule registry "cos" x
  let expCosX := applyNamedRule registry "exp" cosX

  sinXSquared * expCosX

/--
  Example of composing custom rules to build a complex function.
-/
def compositeFunctionExample : IO Unit := do
  -- Define test points
  let xs : List Float := [0.0, 0.5, 1.0, 1.5, 2.0]

  IO.println "Composite Function Example"
  IO.println "-------------------------"
  IO.println "f(x) = sin(x^2) * exp(cos(x))"
  IO.println ""

  for x in xs do
    let result := compositeFunction (seed x)

    -- For verification, compute the function value directly
    let directValue := Float.sin (x * x) * Float.exp (Float.cos x)

    IO.println s!"At x = {x}:"
    IO.println s!"  Function value: {result.primal}"
    IO.println s!"  Direct computation: {directValue}"
    IO.println s!"  Derivative: {result.tangent}"
    IO.println ""

/--
  Run all custom rule examples
-/
def main : IO Unit := do
  customDerivativeExample
  IO.println "--------------------------------\n"
  registryExample
  IO.println "--------------------------------\n"
  compositeFunctionExample

end LeanDidax2.CustomRulesExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.CustomRulesExamples.main
