/-
  Examples of using the LeanDidax automatic differentiation system.
  This file provides concrete examples of both forward and reverse mode autodiff.
-/

import LeanDidax2.Basic
import LeanDidax2.Autodiff

namespace LeanDidax2.Examples

open LeanDidax2
open LeanDidax2.ReverseMode

/--
  A simple polynomial function: f(x) = x^2 + 2*x + 1
-/
def polynomial (x : Value Float) : Value Float :=
  x * x + 2 * x + 1

-- Inline demonstration of forward-mode autodiff with a polynomial function
#eval let x := seed 3.0  -- Create a variable with tangent=1.0
      let result := polynomial x
      -- For polynomial: f(x) = x^2 + 2x + 1, f'(x) = 2x + 2
      -- At x = 3: f'(3) = 2*3 + 2 = 8
      let expected_derivative := 2.0 * 3.0 + 2.0 -- f'(x) = 2x + 2
      (s!"Function value at x=3: {result.primal}",
       s!"Computed derivative: {result.tangent}",
       s!"Expected derivative: {expected_derivative}",
       s!"Correct?: {if Float.abs (result.tangent - expected_derivative) < 1e-10 then "Yes" else "No"}")

/--
  A more complex function: f(x) = x^3 - 2*x^2 + x
-/
def cubicFunction (x : Value Float) : Value Float :=
  x * x * x - 2 * x * x + x

-- Inline demonstration of forward-mode autodiff on cubic function
#eval let x := seed 2.0
      let result := cubicFunction x
      -- For cubic: f(x) = x^3 - 2x^2 + x, f'(x) = 3x^2 - 4x + 1
      -- At x = 2: f'(2) = 3*2^2 - 4*2 + 1 = 12 - 8 + 1 = 5
      let expected_derivative := 3.0 * 2.0^2 - 4.0 * 2.0 + 1.0 -- f'(x) = 3x^2 - 4x + 1
      (s!"Function value at x=2: {result.primal}",
       s!"Computed derivative: {result.tangent}",
       s!"Expected derivative: {expected_derivative}",
       s!"Correct?: {if Float.abs (result.tangent - expected_derivative) < 1e-10 then "Yes" else "No"}")

/--
  A function combining several operations: f(x) = (x + 1) * (x - 2) / (x + 3)
-/
def rationalFunction (x : Value Float) : Value Float :=
  (x + 1) * (x - 2) / (x + 3)

-- Test rational function at x = 4.0
#eval let x := seed 4.0
      let result := rationalFunction x
      let expected_primal := (4.0 + 1.0) * (4.0 - 2.0) / (4.0 + 3.0)
      -- Analytical derivative of f(x) = (x + 1) * (x - 2) / (x + 3)
      -- f'(x) = ((x + 3)((x + 1) + (x - 2)) - (x + 1)(x - 2)) / (x + 3)^2
      let expected_derivative := ((4.0 + 3.0) * ((4.0 + 1.0) + (4.0 - 2.0)) - (4.0 + 1.0) * (4.0 - 2.0)) / ((4.0 + 3.0) * (4.0 + 3.0))
      (s!"Function value at x=4: {result.primal}",
       s!"Expected function value: {expected_primal}",
       s!"Function value correct?: {if Float.abs (result.primal - expected_primal) < 1e-10 then "Yes" else "No"}",
       s!"Computed derivative: {result.tangent}",
       s!"Expected derivative: {expected_derivative}",
       s!"Derivative correct?: {if Float.abs (result.tangent - expected_derivative) < 1e-10 then "Yes" else "No"}")

/--
  A composite function: f(x) = g(h(x)) where g(x) = x^2 and h(x) = 2x + 1
-/
def square (x : Value Float) : Value Float := x * x

def lineFunction (x : Value Float) : Value Float := 2 * x + 1

def compositeFunction (x : Value Float) : Value Float :=
  square (lineFunction x)

-- Test composite function and demonstrate chain rule
#eval let x := seed 3.0
      let intermediate := lineFunction x
      let result := compositeFunction x
      -- h(x) = 2x + 1, h'(x) = 2
      -- g(y) = y^2, g'(y) = 2y
      -- f(x) = g(h(x)), f'(x) = g'(h(x)) * h'(x) = 2h(x) * 2 = 4h(x) = 4(2x + 1)
      let expected_derivative := 4.0 * (2.0 * 3.0 + 1.0)
      (s!"Intermediate value h(3) = {intermediate.primal}",
       s!"Intermediate derivative h'(3) = {intermediate.tangent}",
       s!"Final value f(3) = {result.primal}",
       s!"Computed derivative f'(3) = {result.tangent}",
       s!"Expected derivative f'(3) = {expected_derivative}",
       s!"Chain rule verified?: {if Float.abs (result.tangent - expected_derivative) < 1e-10 then "Yes" else "No"}")

/--
  A function with multiple inputs: f(x, y) = x*y + x^2
-/
def multiInputFunction (x y : Value Float) : Value Float := x * y + x * x

-- Test how to compute partial derivatives with multiple inputs
#eval
  -- For ∂f/∂x at (x=2, y=3), set x.tangent = 1, y.tangent = 0
  let x := seed 2.0
  let y := { primal := 3.0, tangent := 0.0 }
  let result1 := multiInputFunction x y
  -- ∂f/∂x = y + 2x = 3 + 2*2 = 7
  let expected_dx := 3.0 + 2.0 * 2.0

  -- For ∂f/∂y at (x=2, y=3), set x.tangent = 0, y.tangent = 1
  let x2 := { primal := 2.0, tangent := 0.0 }
  let y2 := seed 3.0
  let result2 := multiInputFunction x2 y2
  -- ∂f/∂y = x = 2
  let expected_dy := 2.0

  (s!"Function value at (2,3): {result1.primal}",
   s!"Partial derivative ∂f/∂x: {result1.tangent}",
   s!"Expected ∂f/∂x: {expected_dx}",
   s!"Partial derivative ∂f/∂y: {result2.tangent}",
   s!"Expected ∂f/∂y: {expected_dy}")

/--
  Trigonometric function example: f(x) = sin(x) * cos(x)
  This is equal to (1/2)sin(2x) and its derivative is cos(2x)
-/
def sinCosProduct (x : Value Float) : Value Float := sin x * cos x

-- Pi constant value (approximately 3.14159...)
def pi : Float := 3.14159265358979323846

-- Test trigonometric function differentiation
#eval let x := seed (pi/4)
      let result := sinCosProduct x
      -- f(x) = sin(x) * cos(x) = (1/2)sin(2x)
      -- f'(x) = cos(x) * cos(x) + sin(x) * (-sin(x)) = cos(x)^2 - sin(x)^2 = cos(2x)
      -- We need to explicitly calculate this using the chain rule on sin(x)*cos(x)
      let sinX := Float.sin (pi/4)
      let cosX := Float.cos (pi/4)
      -- The derivative is: cos(x)*cos(x) - sin(x)*sin(x) = cos(x)^2 - sin(x)^2 = cos(2x)
      let expected_derivative := cosX * cosX - sinX * sinX
      (s!"Function value at x=π/4: {result.primal}",
       s!"Computed derivative: {result.tangent}",
       s!"Expected derivative: {expected_derivative}",
       s!"Correct?: {if Float.abs (result.tangent - expected_derivative) < 1e-10 then "Yes" else "No"}")

/--
  End-to-end test: numerical vs automatic differentiation
-/
def endToEndComparisonFloat (f : Float → Float) (df : Value Float → Value Float) (x : Float) : String :=
  -- Forward-mode autodiff
  let xValue := seed x
  let result := df xValue

  -- Numerical differentiation
  let h : Float := 1e-5
  let numDiff := (f (x + h) - f x) / h

  let autodiffResult := result.tangent
  s!"At x = {x}, autodiff derivative: {autodiffResult}, numerical derivative: {numDiff}, difference: {Float.abs (autodiffResult - numDiff)}"

-- Run end-to-end comparison for polynomial function
#eval
  let evalPolynomial (x : Float) : Float := x * x + 2 * x + 1
  endToEndComparisonFloat evalPolynomial polynomial 3.0

-- Run end-to-end comparison for cubic function
#eval
  let evalCubic (x : Float) : Float := x * x * x - 2 * x * x + x
  endToEndComparisonFloat evalCubic cubicFunction 2.0

-- Run end-to-end comparison for rational function
#eval
  let evalRational (x : Float) : Float := (x + 1) * (x - 2) / (x + 3)
  endToEndComparisonFloat evalRational rationalFunction 4.0

-- Run end-to-end comparison for trigonometric function
#eval
  let evalTrig (x : Float) : Float := Float.sin x * Float.cos x
  endToEndComparisonFloat evalTrig sinCosProduct (pi/4)

/--
  Demonstrate forward-mode differentiation on a polynomial function
-/
def forwardModeExample : IO Unit := do
  -- Create an input value
  let inputValue := 3.0
  let input := seed inputValue

  -- Compute f(x) with derivatives
  let output := polynomial input

  -- Print results
  IO.println s!"Function: f(x) = x^2 + 2*x + 1"
  IO.println s!"Input: x = {inputValue}"
  IO.println s!"Output: f({inputValue}) = {output.primal}"
  IO.println s!"Derivative: f'({inputValue}) = {output.tangent}"

  -- Verify with analytical derivative: f'(x) = 2*x + 2
  let analyticalDerivative := 2.0 * inputValue + 2.0
  IO.println s!"Analytical derivative: f'({inputValue}) = {analyticalDerivative}"

/--
  Demonstrate reverse mode autodiff examples using computational graphs
-/
def reverseModeExample : IO Unit := do
  -- Compute derivative of polynomial function at x=3.0
  let x := 3.0
  let expectedGradPoly := 2.0 * x + 2.0

  -- Build the computational graph for polynomial: f(x) = x^2 + 2x + 1
  let xNode := Node.Leaf x
  let x2 := Node.Mul xNode xNode        -- x^2
  let two := Node.Leaf 2.0
  let twoX := Node.Mul two xNode        -- 2*x
  let sum1 := Node.Add x2 twoX          -- x^2 + 2*x
  let one := Node.Leaf 1.0
  let polyGraph := Node.Add sum1 one    -- x^2 + 2*x + 1

  -- Evaluate the graph to get the value
  let polyValue := eval polyGraph

  -- Compute gradients with backward pass
  let gradResults := backward polyGraph 1.0
  let polyGrad := match gradResults.find? (fun pair => pair.1 == x) with
                  | some (_, cotangent) => cotangent
                  | none => 0.0

  IO.println s!"Reverse mode autodiff example:"
  IO.println s!"Polynomial f(x) = x^2 + 2x + 1 at x={x}"
  IO.println s!"Value: {polyValue}"
  IO.println s!"Gradient: {polyGrad}"
  IO.println s!"Expected gradient: {expectedGradPoly}"
  IO.println s!"Correct?: {if Float.abs (polyGrad - expectedGradPoly) < 1e-10 then "Yes" else "No"}"

  -- Compute derivative of cubic function at x=2.0
  let x2 := 2.0
  let expectedGradCubic := 3.0 * x2 * x2 - 4.0 * x2 + 1.0

  -- Build the computational graph for cubic: f(x) = x^3 - 2x^2 + x
  let xNode2 := Node.Leaf x2
  let xSq := Node.Mul xNode2 xNode2     -- x^2
  let xCube := Node.Mul xSq xNode2      -- x^3
  let two := Node.Leaf 2.0
  let twoXSq := Node.Mul two xSq        -- 2x^2
  let diff := Node.Sub xCube twoXSq     -- x^3 - 2x^2
  let cubicGraph := Node.Add diff xNode2 -- x^3 - 2x^2 + x

  -- Evaluate the graph to get the value
  let cubicValue := eval cubicGraph

  -- Compute gradients with backward pass
  let gradResults2 := backward cubicGraph 1.0
  let cubicGrad := match gradResults2.find? (fun pair => pair.1 == x2) with
                   | some (_, cotangent) => cotangent
                   | none => 0.0

  IO.println s!"\nCubic f(x) = x^3 - 2x^2 + x at x={x2}"
  IO.println s!"Value: {cubicValue}"
  IO.println s!"Gradient: {cubicGrad}"
  IO.println s!"Expected gradient: {expectedGradCubic}"
  IO.println s!"Correct?: {if Float.abs (cubicGrad - expectedGradCubic) < 1e-10 then "Yes" else "No"}"

  -- Compute derivative of sin(x)*cos(x) at x=π/4
  let x3 := pi/4
  let sinX := Float.sin x3
  let cosX := Float.cos x3
  let expectedGradTrig := cosX * cosX - sinX * sinX -- cos(2x)

  -- Build the computational graph for sin(x)*cos(x)
  let xNode3 := Node.Leaf x3
  let sinNode := Node.Sin xNode3
  let cosNode := Node.Cos xNode3
  let trigGraph := Node.Mul sinNode cosNode

  -- Evaluate the graph to get the value
  let trigValue := eval trigGraph

  -- Compute gradients with backward pass
  let gradResults3 := backward trigGraph 1.0
  let trigGrad := match gradResults3.find? (fun pair => pair.1 == x3) with
                  | some (_, cotangent) => cotangent
                  | none => 0.0

  IO.println s!"\nTrigonometric f(x) = sin(x)*cos(x) at x={x3}"
  IO.println s!"Value: {trigValue}"
  IO.println s!"Gradient: {trigGrad}"
  IO.println s!"Expected gradient: {expectedGradTrig}"
  IO.println s!"Correct?: {if Float.abs (trigGrad - expectedGradTrig) < 1e-10 then "Yes" else "No"}"

end LeanDidax2.Examples

/-- Main function accessible from the command line -/
def main : IO Unit := do
  IO.println "LeanDidax Forward Mode Autodiff Examples"
  IO.println "--------------------------------------"
  IO.println "This file contains examples in #eval statements that are run during compilation."
  IO.println "Review the file LeanDidax2/Examples.lean to see the examples and their output."
  IO.println ""
  IO.println "When you build the project with 'lake build', the examples will be evaluated"
  IO.println "and their results will be displayed in the build output."
