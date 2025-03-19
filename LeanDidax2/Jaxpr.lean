/-
  Jaxpr: JAX's internal intermediate representation for programs.
  Based on Part 2 of the Autodidax tutorial.
-/

import Lean.Data.Array
import Lean.Data.HashMap
import LeanDidax2.Basic

namespace LeanDidax2.Jaxpr

/-- A variable in a Jaxpr program -/
structure Var where
  name : String
  type : String -- Using String instead of Type for comparison
  deriving Repr, BEq

/-- A literal constant in a Jaxpr program -/
structure Lit where
  value : Float
  deriving Repr, BEq

/-- An equation in a Jaxpr program -/
structure Eqn where
  lhs : Array Var
  rhs : Array (Var ⊕ Lit)
  primitive : String
  deriving Repr, BEq

/-- A complete Jaxpr program -/
structure Jaxpr where
  inputs : Array Var
  equations : Array Eqn
  outputs : Array (Var ⊕ Lit)
  deriving Repr, BEq

instance : ToString Var where
  toString v := s!"Var(name: {v.name}, type: {v.type})"

instance : ToString Lit where
  toString l := s!"Lit(value: {l.value})"

instance : ToString Eqn where
  toString eqn := s!"{eqn.primitive}({eqn.lhs} = {eqn.rhs})"

instance : ToString Jaxpr where
  toString jaxpr := s!"Jaxpr(inputs: {jaxpr.inputs}, equations: {jaxpr.equations}, outputs: {jaxpr.outputs})"


/-- Test suite for Jaxpr functionality -/
def jaxprTests : List (IO Unit) := [
  do
    IO.println "Testing Jaxpr.mk..."
    let x := Var.mk "x" "Float"
    let y := Var.mk "y" "Float"
    let l := Lit.mk 2.0
    let eqn := Eqn.mk #[x] #[Sum.inr l, Sum.inl y] "mul"
    let prog := Jaxpr.mk #[x, y] #[eqn] #[Sum.inl x]
    IO.println s!"Program: {prog}"
    assert! prog.inputs.size = 2
    assert! prog.equations.size = 1
    assert! prog.outputs.size = 1
]

/-- Run all Jaxpr tests -/
def runTests : IO Unit := do
  for test in jaxprTests do
    try
      test
      IO.println "✓ Test passed"
    catch _ =>
      IO.println s!"✗ Test failed"
#eval runTests
end LeanDidax2.Jaxpr

namespace Main

def main : IO Unit := do
  IO.println "Running Jaxpr tests..."
  LeanDidax2.Jaxpr.runTests

end Main
