/-
  Jit: Simplified implementation of JIT compilation for JAX-like operations.
  Based on Part 3 of the Autodidax tutorial.
-/

import Std.Data.HashMap
import LeanDidax2.Basic
import LeanDidax2.Jaxpr

namespace LeanDidax2.Jit

open LeanDidax2
open LeanDidax2.Jaxpr
open Std

/-!
  # Just-in-Time Compilation (JIT)

  This module implements a simplified version of JAX's JIT compilation.

  In this implementation, JIT is implemented as a higher-order primitive
  rather than a full transformation. It works by:

  1. Creating a Jaxpr representation of the function
  2. Evaluating that Jaxpr representation with concrete inputs

  We'll implement both "final style" (on-the-fly processing) and
  "initial style" (staged processing) approaches.
-/

/-- Default Var instance for Inhabited typeclass -/
instance : Inhabited Var where
  default := { name := "", type := "" }

/--
  Environment representing bound variables during evaluation
-/
structure Env where
  vars : Std.HashMap String Float
  deriving Inhabited, Repr

/--
  Initialize an empty environment
-/
def Env.empty : Env := { vars := Std.HashMap.empty }

/--
  Set a variable in the environment
-/
def Env.set (env : Env) (name : String) (value : Float) : Env :=
  { env with vars := env.vars.insert name value }

/--
  Get a variable from the environment
-/
def Env.get? (env : Env) (name : String) : Option Float :=
  env.vars[name]?

/--
  Get a variable from the environment or return a default value
-/
def Env.getD (env : Env) (name : String) (default : Float) : Float :=
  match env.vars[name]? with
  | some val => val
  | none => default

/--
  Evaluate a primitive operation
-/
def evalPrimitive (primName : String) (args : Array Float) : Float :=
  match primName with
  | "add" => args[0]! + args[1]!
  | "mul" => args[0]! * args[1]!
  | "sub" => args[0]! - args[1]!
  | "div" => args[0]! / args[1]!
  | "neg" => -args[0]!
  | "sin" => Float.sin args[0]!
  | "cos" => Float.cos args[0]!
  | "exp" => Float.exp args[0]!
  | "log" => Float.log args[0]!
  | _ => panic! s!"Unknown primitive: {primName}"

/--
  Evaluate one equation in a Jaxpr program
-/
def evalEqn (env : Env) (eqn : Eqn) : Env :=
  -- Extract argument values
  let argVals : Array Float := eqn.rhs.map fun arg =>
    match arg with
    | Sum.inl var => env.getD var.name 0.0
    | Sum.inr lit => lit.value

  -- Evaluate the primitive operation
  let result := evalPrimitive eqn.primitive argVals

  -- Store the result
  let outName := eqn.lhs[0]!.name
  env.set outName result

/--
  Evaluate a Jaxpr program with given input values
-/
def evalJaxpr (jaxpr : Jaxpr) (inputs : Array Float) : Array Float :=
  -- Initialize environment with input values
  let env : Env := Id.run do
    let mut env := Env.empty
    for i in [:jaxpr.inputs.size] do
      env := env.set jaxpr.inputs[i]!.name inputs[i]!
    pure env

  -- Evaluate each equation to update the environment
  let env := jaxpr.equations.foldl evalEqn env

  -- Extract the outputs
  jaxpr.outputs.map fun out =>
    match out with
    | Sum.inl var => env.getD var.name 0.0
    | Sum.inr lit => lit.value

/--
  Final style (on-the-fly) implementation of JIT
  This evaluates the function directly with given inputs
-/
def jitFinalStyle (f : Array Float → Array Float) (inputs : Array Float) : Array Float :=
  f inputs

/--
  Initial style (staged) implementation of JIT
  This creates a Jaxpr representation and then evaluates it
-/
def jitInitialStyle (jaxpr : Jaxpr) (inputs : Array Float) : Array Float :=
  evalJaxpr jaxpr inputs

/--
  A simplified JIT function that takes a Jaxpr and returns a function
  that can be applied to concrete inputs
-/
def jit (jaxpr : Jaxpr) : Array Float → Array Float :=
  fun inputs => evalJaxpr jaxpr inputs

/--
  Test examples to demonstrate JIT functionality
-/
def jitTests : List (IO Unit) := [
  do
    IO.println "Testing Jit evaluation..."

    -- Create a simple jaxpr for f(x, y) = x * y + x
    let x := Var.mk "x" "Float"
    let y := Var.mk "y" "Float"
    let tmp := Var.mk "tmp" "Float"
    let tmp2 := Var.mk "tmp2" "Float"

    let mulEqn := Eqn.mk #[tmp] #[Sum.inl x, Sum.inl y] "mul"
    let addEqn := Eqn.mk #[tmp2] #[Sum.inl tmp, Sum.inl x] "add"

    let prog := Jaxpr.mk #[x, y] #[mulEqn, addEqn] #[Sum.inl tmp2]

    -- Evaluate with concrete inputs
    let inputs := #[2.0, 3.0]  -- x=2, y=3
    let result := jit prog inputs

    -- Expected: 2 * 3 + 2 = 8
    let expected := 8.0
    IO.println s!"JIT result: {result[0]!}"
    IO.println s!"Expected: {expected}"
    assert! Float.abs (result[0]! - expected) < 1e-10
]

/--
  Run all JIT tests
-/
def runTests : IO Unit := do
  for test in jitTests do
    try
      test
      IO.println "✓ Test passed"
    catch e =>
      IO.println s!"✗ Test failed: {e.toString}"

-- Using the direct IO call instead of #eval to avoid 'sorry' related errors
def main : IO Unit := runTests

end LeanDidax2.Jit
