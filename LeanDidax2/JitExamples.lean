/-
  Examples demonstrating JIT compilation functionality in LeanDidax2.

  This file contains examples showing how to use the JIT compilation features
  implemented as part of part 3 of the Autodidax tutorial.
-/

import LeanDidax2.Jit
import LeanDidax2.Jaxpr

namespace LeanDidax2.JitExamples

open LeanDidax2.Jit
open LeanDidax2.Jaxpr

/--
  Example 1: Simple polynomial function with manual Jaxpr creation

  Here we demonstrate creating a Jaxpr for a simple polynomial function: f(x) = x^2 + x + 1
  and evaluating it with JIT.
-/
def example1 : IO Unit := do
  -- Original function
  let f (x : Float) := x * x + x + 1.0

  -- Create a Jaxpr for this function manually
  let x := Var.mk "x" "Float"
  let tmp1 := Var.mk "tmp1" "Float"  -- for x^2
  let tmp2 := Var.mk "tmp2" "Float"  -- for x^2 + x
  let tmp3 := Var.mk "tmp3" "Float"  -- for final result
  let one := Lit.mk 1.0

  -- Equations: x^2, x^2 + x, (x^2 + x) + 1
  let eqn1 := Eqn.mk #[tmp1] #[Sum.inl x, Sum.inl x] "mul"
  let eqn2 := Eqn.mk #[tmp2] #[Sum.inl tmp1, Sum.inl x] "add"
  let eqn3 := Eqn.mk #[tmp3] #[Sum.inl tmp2, Sum.inr one] "add"

  let prog := Jaxpr.mk #[x] #[eqn1, eqn2, eqn3] #[Sum.inl tmp3]

  -- Create a JIT-compiled function
  let jittedF := jit prog

  -- Test values
  let testValues := #[1.0, 2.0, 3.0, 4.0, 5.0]

  IO.println "Example 1: Simple polynomial function with JIT compilation"
  IO.println "Function: f(x) = x^2 + x + 1"
  IO.println "------------------------------------------------------"

  for x in testValues do
    let original := f x
    let jitted := jittedF #[x]
    IO.println s!"x = {x}, Original: {original}, JIT: {jitted[0]!}"

  IO.println ""

/--
  Example 2: More complex function with manual Jaxpr creation

  Here we demonstrate a more complex function: f(x) = x^3 + 2x^2 + 3x + 4
-/
def example2 : IO Unit := do
  -- Original function: f(x) = x^3 + 2x^2 + 3x + 4
  let f (x : Float) := x * x * x + 2.0 * x * x + 3.0 * x + 4.0

  -- Create a Jaxpr manually
  let x := Var.mk "x" "Float"
  let x2 := Var.mk "x2" "Float"        -- x^2
  let x3 := Var.mk "x3" "Float"        -- x^3
  let term1 := Var.mk "term1" "Float"  -- 2x^2
  let term2 := Var.mk "term2" "Float"  -- 3x
  let term3 := Var.mk "term3" "Float"  -- x^3 + 2x^2
  let term4 := Var.mk "term4" "Float"  -- x^3 + 2x^2 + 3x
  let result := Var.mk "result" "Float" -- final result

  -- Literal constants
  let litTwo := Lit.mk 2.0
  let litThree := Lit.mk 3.0
  let litFour := Lit.mk 4.0

  -- Equations
  let eqns := #[
    Eqn.mk #[x2] #[Sum.inl x, Sum.inl x] "mul",              -- x^2
    Eqn.mk #[x3] #[Sum.inl x, Sum.inl x2] "mul",             -- x^3
    Eqn.mk #[term1] #[Sum.inr litTwo, Sum.inl x2] "mul",     -- 2x^2
    Eqn.mk #[term2] #[Sum.inr litThree, Sum.inl x] "mul",    -- 3x
    Eqn.mk #[term3] #[Sum.inl x3, Sum.inl term1] "add",      -- x^3 + 2x^2
    Eqn.mk #[term4] #[Sum.inl term3, Sum.inl term2] "add",   -- x^3 + 2x^2 + 3x
    Eqn.mk #[result] #[Sum.inl term4, Sum.inr litFour] "add" -- x^3 + 2x^2 + 3x + 4
  ]

  let prog := Jaxpr.mk #[x] eqns #[Sum.inl result]

  -- Create JIT-compiled function
  let customJit := jit prog

  -- Test values
  let testValues := #[1.0, 2.0, 3.0]

  IO.println "Example 2: Custom function with JIT compilation"
  IO.println "Function: f(x) = x^3 + 2x^2 + 3x + 4"
  IO.println "------------------------------------------------------"

  for x in testValues do
    let original := f x
    let jitted := customJit #[x]
    IO.println s!"x = {x}, Original: {original}, JIT: {jitted[0]!}"
    IO.println s!"Difference: {Float.abs (original - jitted[0]!)}"

  IO.println ""

/--
  Main function to run all examples
-/
def runExamples : IO Unit := do
  IO.println "=== LeanDidax2: JIT Compilation Examples ==="
  IO.println ""
  IO.println "Part 3 of the Autodidax tutorial implements JIT compilation,"
  IO.println "which speeds up computation by:"
  IO.println "1. Creating a Jaxpr representation of the computation"
  IO.println "2. Executing the Jaxpr with concrete inputs"
  IO.println ""

  example1
  example2

def main : IO Unit := runExamples

end LeanDidax2.JitExamples

#eval LeanDidax2.JitExamples.main
