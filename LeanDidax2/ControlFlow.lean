/-
  Control flow primitives for differentiable programming in LeanDidax.
  Implements JAX-like conditional primitives that can be used in autodiff.
-/

import LeanDidax2.Basic

namespace LeanDidax2

/-
  This module implements differentiable control flow primitives
  similar to those in JAX. These allow for conditional operations
  within a differentiable context.
-/
namespace ControlFlow

/--
  Differentiable conditional operation.
  This is similar to JAX's `lax.cond` function, which allows for
  differentiable branching based on a predicate.

  The gradient will flow through whichever branch is taken.
-/
def cond {α : Type} [Zero α]
  (pred : Bool)
  (trueBranch : Unit → Value α)
  (falseBranch : Unit → Value α) : Value α :=
  if pred then
    trueBranch ()
  else
    falseBranch ()

/--
  Differentiable `where` operation.
  This selects elements from `onTrue` or `onFalse` based on the predicate.
  Similar to JAX's `lax.select`.

  Simpler than `cond` - just selects values based on condition.
-/
def select {α : Type} [Zero α]
  (pred : Bool)
  (onTrue : Value α)
  (onFalse : Value α) : Value α :=
  if pred then onTrue else onFalse

/--
  A switch-like conditional that selects one of several branches based on an index.
  Similar to JAX's `lax.switch`.

  The gradient will flow through the selected branch.

  This is a basic implementation that only works with a small number of branches.
-/
def switch {α : Type} [Zero α]
  (index : Nat)
  (branches : List (Unit → Value α)) : Value α :=
  match branches[index]? with
  | some branch => branch ()
  | none => match branches[0]? with
    | some firstBranch => firstBranch () -- Default to first branch if index out of bounds
    | none => { primal := 0, tangent := 0 } -- Default value if no branches

/--
  A more flexible/general way to implement differentiable branching.
  This takes a function that produces a branch function based on some value,
  and ensures that gradients flow through correctly.
-/
def branch {α β : Type} [Zero β]
  (branchVal : α)
  (branchFn : α → (Unit → Value β)) : Value β :=
  (branchFn branchVal) ()

/--
  Demonstrates the use of control flow primitives with a simple example.
-/
def conditionalFunction (x : Value Float) : Value Float :=
  cond (x.primal > 0.0)
    (fun _ => x * x) -- x^2 when x > 0
    (fun _ => x * (const (-1.0))) -- -x when x ≤ 0

/--
  A demonstration of custom control flow with piecewise functions.
  This implements relu(x) = max(0, x) in a differentiable way.
-/
def reluWithControlFlow (x : Value Float) : Value Float :=
  select (x.primal > 0.0) x (const 0.0)

/--
  A more complex example with differentiable switch
-/
def switchExample (x : Value Float) (index : Nat) : Value Float :=
  switch index [
    (fun _ => x * x),            -- x^2
    (fun _ => x * x * x),        -- x^3
    (fun _ => x + (const 5.0)),  -- x + 5
    (fun _ => sin x),            -- sin(x)
    (fun _ => exp x)             -- e^x
  ]

end ControlFlow

end LeanDidax2
