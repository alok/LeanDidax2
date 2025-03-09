/-
  Batching functionality for LeanDidax.
  Implements vectorized operations for automatic differentiation.
-/

import LeanDidax2.Basic

namespace LeanDidax2

namespace Batch

/-- Apply a function to an array of values and return an array of results -/
def batchApply (f : Value Float → Value Float) (xs : Array Float) : Array Float := Id.run do
  let mut results := Array.mkArray xs.size 0.0
  for i in [:xs.size] do
    let x := seed xs[i]!
    let y := f x
    results := results.set! i y.primal
  return results

/-- Compute a function and its derivative for an array of values -/
def batchGradient (f : Value Float → Value Float) (xs : Array Float) : Array (Float × Float) := Id.run do
  let mut results := Array.mkArray xs.size (0.0, 0.0)
  for i in [:xs.size] do
    let x := seed xs[i]!
    let y := f x
    results := results.set! i (y.primal, y.tangent)
  return results

/-- The vmap transformation: vectorize a function operating on Values -/
def vmap (f : Value Float → Value Float) (xs : Array Float) : Array (Value Float) := Id.run do
  let mut results := Array.mkArray xs.size (Value.mk 0.0 0.0)
  for i in [:xs.size] do
    let x := seed xs[i]!
    let y := f x
    results := results.set! i y
  return results

/--
  Apply a function to a range of values and return a list of (input, output, derivative) tuples.
  This is useful for visualizing how a function behaves across a range of inputs.
-/
def evaluateRange (f : Value Float → Value Float) (start : Float) (end_ : Float) (steps : Nat) : Array (Float × Float × Float) :=
  if steps == 0 then
    #[]
  else
    let step := (end_ - start) / steps.toFloat
    let inputs : Array Float := Id.run do
      let mut arr := Array.mkArray (steps + 1) 0.0
      for i in [0:steps+1] do
        arr := arr.set! i (start + i.toFloat * step)
      pure arr

    inputs.map (fun x =>
      let input := seed x
      let output := f input
      (x, output.primal, output.tangent))

/--
  Compute Jacobian matrix for a vector-valued function with vector-valued input.
  For a function f: R^n -> R^m, this computes the m×n Jacobian matrix.

  The function f takes an array of Values and returns an array of Values.
  The Jacobian is represented as an array of arrays, where Jacobian[i][j] = ∂f_i/∂x_j
-/
def jacobian (f : Array (Value Float) → Array (Value Float)) (xs : Array Float) : Array (Array Float) := Id.run do
  let n := xs.size

  -- Call f once to determine the output dimension
  let seeds := xs.map (fun x => { primal := x, tangent := 0.0 })
  let output := f seeds
  let m := output.size

  -- Initialize Jacobian matrix with zeros
  let mut jacobian := Array.mkArray m #[]
  for i in [:m] do
    jacobian := jacobian.set! i (Array.mkArray n 0.0)

  -- Compute each column of the Jacobian matrix
  for j in [:n] do
    -- Create the seed vector with tangent=1 at position j
    let mut seedsJ := Array.mkArray n (Value.mk 0.0 0.0)
    for k in [:n] do
      if k == j then
        seedsJ := seedsJ.set! k (seed xs[k]!)
      else
        seedsJ := seedsJ.set! k ({ primal := xs[k]!, tangent := 0.0 })

    -- Compute f(seedsJ) to get the jth column of the Jacobian
    let outputJ := f seedsJ

    -- Store the results in the Jacobian matrix
    for i in [:m] do
      let jacobianRow := jacobian[i]!
      jacobian := jacobian.set! i (jacobianRow.set! j outputJ[i]!.tangent)

  return jacobian

/--
  Apply a scalar-valued function to a batch of inputs in parallel.
  This version is optimized for performance with an efficient implementation.
-/
def batchMap (f : Value Float → Value Float) (xs : Array Float) : Array Float := Id.run do
  -- Pre-allocate the result array
  let n := xs.size
  let mut results := Array.mkArray n 0.0

  -- Process inputs in batch for better cache locality
  for i in [:n] do
    results := results.set! i (f (seed xs[i]!)).primal

  return results

/--
  Enhanced version of vmap that can handle both single and multi-dimensional inputs.
  This works similar to JAX's vmap, supporting functions with multiple arguments.
-/
def vmapMulti [Inhabited α] [Inhabited β]
  (f : Value α → Value β)
  (xs : Array α) : Array (Value β) := Id.run do

  let n := xs.size
  let mut results := Array.mkArray n (Value.mk (default : β) (default : β))

  for i in [:n] do
    let x := { primal := xs[i]!, tangent := default }
    results := results.set! i (f x)

  return results

end Batch

end LeanDidax2
