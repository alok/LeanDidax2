/-
  Examples of using the monad transformers from LeanDidax2.MonadTransformer.
  This file demonstrates how to compose differentiable transformations
  in a JAX-like style.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.MonadTransformer
import LeanDidax2.Batch
import LeanDidax2.Utils

namespace LeanDidax2.MonadTransformerExamples

open LeanDidax2
open LeanDidax2.Monadic
open LeanDidax2.MonadTransformer
open LeanDidax2.Utils

/-- Helper function for numerical literals in DiffM -/
def float2diffm (x : Float) : DiffM Float :=
  { run := fun _ => { primal := x, tangent := 0.0 } }

/--
  A simple polynomial function using the DiffM monad: f(x) = x^2 + 2*x + 1
-/
def polynomialM (x : DiffM Float) : DiffM Float := do
  let x_squared := x * x
  let two_x := float2diffm 2.0 * x
  let sum := x_squared + two_x
  sum + float2diffm 1.0

/--
  A more complex function using the DiffM monad: f(x) = x^3 - 2*x^2 + x
-/
def cubicFunctionM (x : DiffM Float) : DiffM Float := do
  let x_squared := x * x
  let x_cubed := x_squared * x
  let two_x_squared := float2diffm 2.0 * x_squared
  let diff := x_cubed - two_x_squared
  diff + x

/-- Lift a normal DiffM function to work in a DiffReaderT context -/
def liftPolynomial {ρ : Type} (x : DiffReaderT ρ Float) : DiffReaderT ρ Float :=
  withEnv polynomialM x

/-- Lift a normal DiffM function to work in a DiffReaderT context -/
def liftCubic {ρ : Type} (x : DiffReaderT ρ Float) : DiffReaderT ρ Float :=
  withEnv cubicFunctionM x

/--
  Example of using configuration with ReaderT to customize autodiff
-/
def adWithConfigExample : IO Unit := do
  -- Define a function that uses the configuration
  let configuredFunction : DiffReaderT ADConfig Float := do
    let config ← ask
    let x_val := if config.forwardMode then 3.0 else 3.0 * config.seed
    -- Use monadLift directly with differentiableVar
    let x : DiffReaderT ADConfig Float := monadLift (differentiableVar x_val)
    liftPolynomial x

  -- Run with different configurations
  let defaultConfig : ADConfig := {}
  let customConfig : ADConfig := { seed := 2.0, forwardMode := false }

  let defaultResult := withConfig defaultConfig configuredFunction
  let customResult := withConfig customConfig configuredFunction

  IO.println "Using Configuration with ReaderT Example"
  IO.println "--------------------------------------"
  IO.println s!"With default config: {runDiff defaultResult}"
  IO.println s!"With custom config:  {runDiff customResult}"

/--
  Example of using transformed functions with metadata
-/
def transformedFunctionExample : IO Unit := do
  -- Create transformed functions
  let gradPoly := gradTransform polynomialM
  let vmapCubic := vmapTransform (fun x => x * x * x)

  -- Compose transformations using the vectorized version of gradTransform
  let gradFloatToFloat := gradTransform cubicFunctionM
  let mapGrad := mapArrayTransform gradFloatToFloat
  let addOne := vmapTransform (fun x => x + 1.0)
  let composed := composeTransformsBetter addOne mapGrad

  IO.println "Transformed Functions with Metadata Example"
  IO.println "----------------------------------------"

  -- Print transformation metadata
  IO.println s!"gradPoly transformations: {gradPoly.transformations}"
  IO.println s!"vmapCubic transformations: {vmapCubic.transformations}"
  IO.println s!"composed transformations: {composed.transformations}"

  -- Apply the transformed functions
  let x := 2.0
  let inputs := #[1.0, 2.0, 3.0]

  IO.println s!"\ngradPoly({x}) = {gradPoly.apply x}"
  IO.println s!"vmapCubic({inputs}) = {vmapCubic.apply inputs}"

  -- Apply the composed transformation
  let gradResults := composed.apply inputs
  IO.println s!"composed({inputs}) = {gradResults}"

  -- The expected result for x=3 would be:
  -- grad(x^3 - 2x^2 + x)(3) = 3*3^2 - 4*3 + 1 = 27 - 12 + 1 = 16
  -- Then add 1 to get 17
  let expected3 := 16.0 + 1.0

  -- Check the result for the third element (index 2)
  if inputs.size > 2 then
    IO.println s!"For x=3: Expected: {expected3}, Got: {gradResults[2]!}"
    IO.println s!"Match: {if Float.abs (gradResults[2]! - expected3) < 1e-10 then "Yes" else "No"}"

/--
  Example of using batch gradient computation
-/
def batchGradientExample : IO Unit := do
  -- Create an array of inputs
  let inputs : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0]

  -- Compute gradients for all inputs at once
  let gradResults := batchGrad polynomialM inputs

  IO.println "Batch Gradient Example"
  IO.println "--------------------"
  IO.println "f(x) = x^2 + 2x + 1, f'(x) = 2x + 2"

  for i in [:inputs.size] do
    let x := inputs[i]!
    let expectedGrad := 2.0 * x + 2.0
    IO.println s!"grad(f) at x = {x}: {gradResults[i]!}"
    IO.println s!"Expected: {expectedGrad}"
    IO.println s!"Correct?: {if Float.abs (gradResults[i]! - expectedGrad) < 1e-10 then "Yes" else "No"}"
    IO.println ""

  -- Compute both values and gradients
  let valueAndGradResults := batchValueAndGrad polynomialM inputs

  IO.println "Value and Gradient Results:"
  for i in [:inputs.size] do
    let x := inputs[i]!
    let (value, gradient) := valueAndGradResults[i]!
    let expectedValue := x * x + 2.0 * x + 1.0
    let expectedGrad := 2.0 * x + 2.0
    IO.println s!"At x = {x}: value = {value}, gradient = {gradient}"
    IO.println s!"Expected: value = {expectedValue}, gradient = {expectedGrad}"
    IO.println s!"Correct?: {if Float.abs (value - expectedValue) < 1e-10 &&
                               Float.abs (gradient - expectedGrad) < 1e-10
                            then "Yes" else "No"}"
    IO.println ""

/--
  Example of using withADConfig to pass configuration to a computation
-/
def configurationExample : IO Unit := do
  -- Define a computation that uses configuration
  let computation : DiffReaderT ADConfig Float := do
    let config ← ask
    let scale := if config.forwardMode then 1.0 else config.seed
    let x_val := 2.0 * scale
    -- Simplify using monadLift
    let x := monadLift (differentiableVar x_val)
    liftPolynomial x

  -- Run with different configurations
  let config1 : ADConfig := {}  -- default config
  let config2 : ADConfig := { seed := 2.0 }
  let config3 : ADConfig := { forwardMode := false, seed := 3.0 }

  let result1 := runDiff (withADConfig config1 computation)
  let result2 := runDiff (withADConfig config2 computation)
  let result3 := runDiff (withADConfig config3 computation)

  IO.println "Configuration with withADConfig Example"
  IO.println "------------------------------------"
  IO.println s!"Result with default config: {result1}"
  IO.println s!"Result with seed=2.0: {result2}"
  IO.println s!"Result with forwardMode=false, seed=3.0: {result3}"

/--
  Example of debugging a transformation using the debugTransform utility
-/
def debuggingExample : IO Unit := do
  -- Create a simple function
  let square (x : Float) := x * x

  -- Add debugging wrapper
  let debugSquare := debugTransform "square" square

  -- Use in a transformation pipeline
  let debugTransformation := vmapTransform debugSquare

  -- Create a version with debug config
  let config : ADConfig := { debug := true }
  let debugConfigSquare := configuredTransform "square_with_config" square config

  IO.println "Debugging Transformations Example"
  IO.println "------------------------------"
  IO.println "Running with debug output to trace:"

  -- Apply the transformation (debug output will go to trace)
  let inputs := #[1.0, 2.0, 3.0]
  let _ := debugTransformation.apply inputs
  let _ := debugConfigSquare.apply 4.0

  IO.println "Transformation complete (check trace output)"

/--
  Example using the new liftTransform utility
-/
def liftTransformExample : IO Unit := do
  -- Define a simple function that returns a DiffM
  let f (x : Float) : DiffM Float :=
    differentiableVar (x * x)

  -- Create a DiffReaderT computation that uses liftTransform
  let computation : DiffReaderT ADConfig Float := do
    let config ← ask
    let x := if config.forwardMode then 3.0 else 3.0 * config.seed

    -- Use liftTransform to lift f to DiffReaderT
    -- This lifts the entire DiffM computation into the DiffReaderT context
    liftTransform f x

  let config1 : ADConfig := {}
  let config2 : ADConfig := { seed := 2.0, forwardMode := false }

  let result1 := runDiff (withADConfig config1 computation)
  let result2 := runDiff (withADConfig config2 computation)

  -- For comparison, compute what we expect directly
  let expected1 := 3.0 * 3.0  -- With default config, x = 3.0
  let expected2 := (3.0 * 2.0) * (3.0 * 2.0)  -- With custom config, x = 3.0 * 2.0 = 6.0

  IO.println "Lift Transform Example"
  IO.println "--------------------"
  IO.println s!"Result with default config: {result1}"
  IO.println s!"Expected: {expected1}"

  IO.println s!"Result with custom config: {result2}"
  IO.println s!"Expected: {expected2}"

/--
  Run all monad transformer examples
-/
def main : IO Unit := do
  adWithConfigExample
  IO.println "\n"
  transformedFunctionExample
  IO.println "\n"
  batchGradientExample
  IO.println "\n"
  configurationExample
  IO.println "\n"
  debuggingExample
  IO.println "\n"
  liftTransformExample

end LeanDidax2.MonadTransformerExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.MonadTransformerExamples.main
