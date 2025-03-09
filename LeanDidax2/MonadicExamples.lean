/-
  Examples of using the monadic automatic differentiation system in LeanDidax.
  This file demonstrates the elegance and composability of the monadic approach.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic

namespace LeanDidax2.MonadicExamples

open LeanDidax2
open LeanDidax2.Monadic

/-- Create a constant DiffM value -/
def constant (x : Float) : DiffM Float := {
  run := fun _ => { primal := x, tangent := 0 }
}

/--
  A simple polynomial function using monadic style: f(x) = x^2 + 2*x + 1
-/
def polynomialM (x : DiffM Float) : DiffM Float := do
  let x_squared := mul x x
  let two := constant 2.0
  let two_x := mul two x
  let sum := add x_squared two_x
  add sum (constant 1.0)

/--
  A more complex function using monadic style: f(x) = x^3 - 2*x^2 + x
-/
def cubicFunctionM (x : DiffM Float) : DiffM Float := do
  let x_squared := mul x x
  let x_cubed := mul x_squared x
  let two := constant 2.0
  let two_x_squared := mul two x_squared
  let diff := sub x_cubed two_x_squared
  add diff x

/--
  A rational function using monadic style: f(x) = (x + 1) * (x - 2) / (x + 3)
-/
def rationalFunctionM (x : DiffM Float) : DiffM Float := do
  let one := constant 1.0
  let two := constant 2.0
  let three := constant 3.0
  let num1 := add x one
  let num2 := sub x two
  let den := add x three
  let num := mul num1 num2
  div num den

/--
  A composite function using monadic style: f(x) = g(h(x))
  where g(x) = x^2 and h(x) = 2x + 1
-/
def squareM (x : DiffM Float) : DiffM Float :=
  mul x x

def lineFunctionM (x : DiffM Float) : DiffM Float := do
  let two := constant 2.0
  let one := constant 1.0
  let two_x := mul two x
  add two_x one

def compositeFunctionM (x : DiffM Float) : DiffM Float := do
  let h_x := lineFunctionM x
  squareM h_x

/--
  A function with multiple inputs using monadic style: f(x, y) = x*y + x^2
-/
def multiInputFunctionM (x y : DiffM Float) : DiffM Float := do
  let xy := mul x y
  let x_squared := mul x x
  add xy x_squared

/--
  A trigonometric function using monadic style: f(x) = sin(x) * cos(x)
-/
def sinCosProductM (x : DiffM Float) : DiffM Float := do
  let sin_x := Monadic.sin x
  let cos_x := Monadic.cos x
  mul sin_x cos_x

/--
  Example of running monadic computations and getting results
-/
def monadicExample : IO Unit := do
  -- Define input values
  let x1 := 3.0
  let x2 := 2.0
  let x3 := 4.0
  let x4 := Float.pi / 4.0

  -- Compute gradients using monadic approach
  let grad1 := grad polynomialM x1
  let grad2 := grad cubicFunctionM x2
  let grad3 := grad rationalFunctionM x3
  let grad4 := grad sinCosProductM x4

  -- Expected gradients (analytical derivatives)
  let expected1 := 2.0 * x1 + 2.0  -- f'(x) = 2x + 2
  let expected2 := 3.0 * x2^2 - 4.0 * x2 + 1.0  -- f'(x) = 3x^2 - 4x + 1
  let expected3 := ((x3 + 3.0) * ((x3 + 1.0) + (x3 - 2.0)) - (x3 + 1.0) * (x3 - 2.0)) / ((x3 + 3.0) * (x3 + 3.0))
  let expected4 := Float.cos(x4)^2 - Float.sin(x4)^2  -- cos(2x)

  -- Print results
  IO.println "Monadic Automatic Differentiation Examples"
  IO.println "----------------------------------------"

  IO.println s!"Polynomial function: f(x) = x^2 + 2x + 1"
  IO.println s!"Gradient at x = {x1}: {grad1}"
  IO.println s!"Expected: {expected1}"
  IO.println s!"Correct?: {if Float.abs (grad1 - expected1) < 1e-10 then "Yes" else "No"}"
  IO.println ""

  IO.println s!"Cubic function: f(x) = x^3 - 2x^2 + x"
  IO.println s!"Gradient at x = {x2}: {grad2}"
  IO.println s!"Expected: {expected2}"
  IO.println s!"Correct?: {if Float.abs (grad2 - expected2) < 1e-10 then "Yes" else "No"}"
  IO.println ""

  IO.println s!"Rational function: f(x) = (x + 1)(x - 2)/(x + 3)"
  IO.println s!"Gradient at x = {x3}: {grad3}"
  IO.println s!"Expected: {expected3}"
  IO.println s!"Correct?: {if Float.abs (grad3 - expected3) < 1e-10 then "Yes" else "No"}"
  IO.println ""

  IO.println s!"Trigonometric function: f(x) = sin(x) * cos(x)"
  IO.println s!"Gradient at x = π/4: {grad4}"
  IO.println s!"Expected: {expected4}"
  IO.println s!"Correct?: {if Float.abs (grad4 - expected4) < 1e-10 then "Yes" else "No"}"
  IO.println ""

  -- Demonstrate the composite function and chain rule
  let (compVal, compGrad) := valueAndGrad compositeFunctionM x1
  let expected_comp := 4.0 * (2.0 * x1 + 1.0)  -- 4(2x + 1)

  IO.println s!"Composite function: f(x) = (2x + 1)^2"
  IO.println s!"Value at x = {x1}: {compVal}"
  IO.println s!"Gradient at x = {x1}: {compGrad}"
  IO.println s!"Expected gradient: {expected_comp}"
  IO.println s!"Correct?: {if Float.abs (compGrad - expected_comp) < 1e-10 then "Yes" else "No"}"
  IO.println ""

  -- Higher-order functions demonstration
  let liftGrad := fun x => lift (Value.mk (grad polynomialM x.run().primal) 0.0)
  let doubleGrad := grad liftGrad x1

  IO.println s!"Second derivative of f(x) = x^2 + 2x + 1"
  IO.println s!"d²f/dx² at x = {x1}: {doubleGrad}"
  IO.println s!"Expected: 2.0 (constant second derivative)"
  IO.println s!"Correct?: {if Float.abs (doubleGrad - 2.0) < 1e-10 then "Yes" else "No"}"

/--
  Example of composing higher-order differentiable functions
-/
def higherOrderExample : IO Unit := do
  -- Define some functions to compose
  let f (x : DiffM Float) : DiffM Float := do
    let x_squared := mul x x
    add x_squared (constant 1.0)

  let g (x : DiffM Float) : DiffM Float := Monadic.sin x
  let h (x : DiffM Float) : DiffM Float := Monadic.exp x

  -- Compose functions in different ways
  let fg (x : DiffM Float) : DiffM Float := do   -- (f ∘ g)(x) = f(g(x)) = sin²(x) + 1
    let gx := g x
    f gx

  let gf (x : DiffM Float) : DiffM Float := do   -- (g ∘ f)(x) = g(f(x)) = sin(x² + 1)
    let fx := f x
    g fx

  let fgh (x : DiffM Float) : DiffM Float := do  -- (f ∘ g ∘ h)(x) = f(g(h(x))) = sin²(e^x) + 1
    let hx := h x
    let ghx := g hx
    f ghx

  -- Compute values and gradients at x = 1.0
  let x := 1.0

  let (fg_val, fg_grad) := valueAndGrad fg x
  let (gf_val, gf_grad) := valueAndGrad gf x
  let (fgh_val, fgh_grad) := valueAndGrad fgh x

  -- Expected values and gradients (verified analytically)
  let fg_expected_val := Float.sin(x)^2 + 1.0
  let fg_expected_grad := 2.0 * Float.sin(x) * Float.cos(x)

  let gf_expected_val := Float.sin(x^2 + 1.0)
  let gf_expected_grad := Float.cos(x^2 + 1.0) * 2.0 * x

  let fgh_expected_val := Float.sin(Float.exp(x))^2 + 1.0
  let fgh_expected_grad := 2.0 * Float.sin(Float.exp(x)) * Float.cos(Float.exp(x)) * Float.exp(x)

  -- Print results
  IO.println "Higher-Order Function Composition Examples"
  IO.println "----------------------------------------"

  IO.println s!"f(x) = x² + 1"
  IO.println s!"g(x) = sin(x)"
  IO.println s!"h(x) = e^x"
  IO.println ""

  IO.println s!"(f ∘ g)(x) = sin²(x) + 1 at x = {x}:"
  IO.println s!"Value: {fg_val}, Expected: {fg_expected_val}"
  IO.println s!"Gradient: {fg_grad}, Expected: {fg_expected_grad}"
  IO.println s!"Correct?: {if Float.abs (fg_val - fg_expected_val) < 1e-10 &&
                               Float.abs (fg_grad - fg_expected_grad) < 1e-10
                            then "Yes" else "No"}"
  IO.println ""

  IO.println s!"(g ∘ f)(x) = sin(x² + 1) at x = {x}:"
  IO.println s!"Value: {gf_val}, Expected: {gf_expected_val}"
  IO.println s!"Gradient: {gf_grad}, Expected: {gf_expected_grad}"
  IO.println s!"Correct?: {if Float.abs (gf_val - gf_expected_val) < 1e-10 &&
                               Float.abs (gf_grad - gf_expected_grad) < 1e-10
                            then "Yes" else "No"}"
  IO.println ""

  IO.println s!"(f ∘ g ∘ h)(x) = sin²(e^x) + 1 at x = {x}:"
  IO.println s!"Value: {fgh_val}, Expected: {fgh_expected_val}"
  IO.println s!"Gradient: {fgh_grad}, Expected: {fgh_expected_grad}"
  IO.println s!"Correct?: {if Float.abs (fgh_val - fgh_expected_val) < 1e-10 &&
                               Float.abs (fgh_grad - fgh_expected_grad) < 1e-10
                            then "Yes" else "No"}"

/--
  Run all monadic examples
-/
def main : IO Unit := do
  monadicExample
  IO.println "\n"
  higherOrderExample

end LeanDidax2.MonadicExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.MonadicExamples.main
