/-
  Custom derivative rules for LeanDidax.
  This implements functionality for defining custom derivatives for functions.
-/

import LeanDidax2.Basic

namespace LeanDidax2

/-
  This module provides functionality for defining and using custom
  derivative rules in the LeanDidax automatic differentiation system.
-/
namespace CustomRules

/--
  A custom differentiation rule for a function.
  This defines both the primal computation and how to compute its derivative.
-/
structure CustomRule (α β : Type) where
  /-- The primal function (regular function evaluation) -/
  primal : α → β
  /-- The derivative function (computing gradient) -/
  derivative : α → β

/--
  Apply a custom rule to a Value, computing both the function and its derivative.
-/
def applyRule {α β : Type} [HMul β α β] (rule : CustomRule α β) (x : Value α) : Value β :=
  { primal := rule.primal x.primal,
    tangent := rule.derivative x.primal * x.tangent }

/--
  A registry of custom rules for different functions.
  This is a simple implementation using a list of (name, rule) pairs.
-/
def Registry (α β : Type) := List (String × CustomRule α β)

/--
  Create an empty registry.
-/
def emptyRegistry {α β : Type} : Registry α β := []

instance : EmptyCollection (Registry α β) where
  emptyCollection := emptyRegistry

/--
  Register a new custom rule.
-/
def registerRule {α β : Type} (registry : Registry α β) (name : String) (rule : CustomRule α β) : Registry α β :=
  (name, rule) :: registry

/--
  Look up a custom rule by name.
-/
def lookupRule {α β : Type} (registry : Registry α β) (name : String) : Option (CustomRule α β) :=
  match registry.find? (fun pair => pair.1 == name) with
  | some (_, rule) => some rule
  | none => none

/--
  Example registry for common mathematical functions on Float.
-/
def floatRegistry : Registry Float Float :=
  let registry : Registry Float Float := []

  -- Custom rule for x^2
  let square : CustomRule Float Float := {
    primal := fun x => x * x,
    derivative := fun x => 2 * x
  }

  -- Custom rule for e^x
  let exp : CustomRule Float Float := {
    primal := Float.exp,
    derivative := Float.exp
  }

  -- Custom rule for ln(x)
  let log : CustomRule Float Float := {
    primal := Float.log,
    derivative := fun x => 1.0 / x
  }

  -- Custom rule for sin(x)
  let sin : CustomRule Float Float := {
    primal := Float.sin,
    derivative := Float.cos
  }

  -- Custom rule for cos(x)
  let cos : CustomRule Float Float := {
    primal := Float.cos,
    derivative := fun x => -Float.sin x
  }

  -- Register all rules
  let registry := registerRule registry "square" square
  let registry := registerRule registry "exp" exp
  let registry := registerRule registry "log" log
  let registry := registerRule registry "sin" sin
  let registry := registerRule registry "cos" cos

  registry

/--
  Apply a named custom rule from the registry.
-/
def applyNamedRule (registry : Registry Float Float) (name : String) (x : Value Float) : Value Float :=
  match lookupRule registry name with
  | some rule => applyRule rule x
  | none => { primal := x.primal, tangent := x.tangent } -- Identity function as fallback

/--
  Define a new function with a custom derivative rule.
  The name parameter is optional and can be used for documentation or future registry integration.
-/
def defCustomFn (_ : String) (f : Float → Float) (df : Float → Float) (x : Value Float) : Value Float :=
  let rule : CustomRule Float Float := {
    primal := f,
    derivative := df
  }
  applyRule rule x

end CustomRules

end LeanDidax2
