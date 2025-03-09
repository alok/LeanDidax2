/-
  ExceptT Adapter for LeanDidax.
  This module provides adapters for using Lean's built-in ExceptT monad transformer
  with differentiable computations, allowing error handling in autodiff.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.MonadTransformer
import LeanDidax2.StateTransformer
import LeanDidax2.Utils

namespace LeanDidax2

namespace ExceptTransformer

/--
  The DiffError type represents possible errors that can occur during
  automatic differentiation.
-/
inductive DiffError
  /-- Domain error, e.g., log of negative number -/
  | domainError (message : String)
  /-- Numerical error, e.g., division by zero -/
  | numericalError (message : String)
  /-- Computation limit exceeded -/
  | computationLimitExceeded (message : String)
  /-- Incompatible shapes in vectorized operations -/
  | shapeMismatch (message : String)
  /-- Other errors -/
  | other (message : String)
  deriving Inhabited, BEq

instance : ToString DiffError where
  toString
    | DiffError.domainError msg => s!"Domain error: {msg}"
    | DiffError.numericalError msg => s!"Numerical error: {msg}"
    | DiffError.computationLimitExceeded msg => s!"Computation limit exceeded: {msg}"
    | DiffError.shapeMismatch msg => s!"Shape mismatch: {msg}"
    | DiffError.other msg => s!"Error: {msg}"

/--
  The DiffExceptT type is an abbreviation for ExceptT with DiffError and DiffM.
  This allows handling errors in differentiable computations.
-/
abbrev DiffExceptT (α : Type) := ExceptT DiffError Monadic.DiffM α

/-- Run a DiffExceptT computation and get an Except result inside DiffM -/
def runDiffExcept {α : Type} (m : DiffExceptT α) : Monadic.DiffM (Except DiffError α) :=
  ExceptT.run m

/-- Safely execute a computation that might cause domain errors -/
def catchDomain {α : Type} (m : DiffExceptT α) (handler : String → DiffExceptT α) : DiffExceptT α := do
  match ← ExceptT.run m with
  | Except.ok a => return a
  | Except.error (DiffError.domainError msg) => handler msg
  | Except.error e => throw e

/-- Safely execute a computation that might cause numerical errors -/
def catchNumerical {α : Type} (m : DiffExceptT α) (handler : String → DiffExceptT α) : DiffExceptT α := do
  match ← ExceptT.run m with
  | Except.ok a => return a
  | Except.error (DiffError.numericalError msg) => handler msg
  | Except.error e => throw e

/-- Apply error checking to a differentiable function -/
def withErrorChecking {α : Type} (m : Monadic.DiffM α) (validate : α → Option DiffError) : DiffExceptT α := do
  let result ← ExceptT.lift m
  match validate result with
  | none => pure result
  | some err => throw err

/-- Safe version of log function that handles domain errors -/
def safelog (x : Value Float) : DiffExceptT (Value Float) := do
  if x.primal <= 0 then
    throw (DiffError.domainError s!"logarithm of non-positive number: {x.primal}")
  else
    ExceptT.lift (pure (LeanDidax2.log x))

/-- Safe version of division that handles division by zero -/
def safediv (num den : Value Float) : DiffExceptT (Value Float) := do
  if den.primal == 0 then
    throw (DiffError.numericalError "division by zero")
  else
    ExceptT.lift (pure (LeanDidax2.div num den))

/-- Safe version of sqrt that handles negative numbers -/
def safesqrt (x : Value Float) : DiffExceptT (Value Float) := do
  if x.primal < 0 then
    throw (DiffError.domainError s!"square root of negative number: {x.primal}")
  else
    -- Define the square root operation with correct derivative
    let result := {
      primal := Float.sqrt x.primal,
      tangent := if x.primal == 0 then 0 else x.tangent / (2.0 * Float.sqrt x.primal)
    }
    pure result

/-- Combine ExceptT with StateT for stateful error handling -/
abbrev DiffExceptStateT (σ : Type) (α : Type) := StateT σ (ExceptT DiffError Monadic.DiffM) α

/-- Run a combined ExceptT and StateT computation -/
def runDiffExceptState {σ α : Type} (s : σ) (m : DiffExceptStateT σ α) :
    Monadic.DiffM (Except DiffError (α × σ)) :=
  ExceptT.run (StateT.run m s)

/--
  Execute a DiffExceptT computation and then run an IO action with the result.
  This handles both successful and error cases.
-/
def runDiffExceptIO {α β : Type}
  (m : DiffExceptT α)
  (onSuccess : α → IO β)
  (onError : DiffError → IO β) : IO β := do
  let diffResult := runDiffExcept m
  let result := Monadic.runDiff diffResult
  match result.primal with
  | Except.ok a => onSuccess a
  | Except.error e => onError e

/--
  Execute a DiffExceptStateT computation, handling both state and errors in IO.
-/
def runDiffExceptStateIO {σ α β : Type}
  (s : σ)
  (m : DiffExceptStateT σ α)
  (onSuccess : α → σ → IO β)
  (onError : DiffError → IO β) : IO β := do
  let diffResult := runDiffExceptState s m
  let result := Monadic.runDiff diffResult
  match result.primal with
  | Except.ok (a, s') => onSuccess a s'
  | Except.error e => onError e

/-- Try a computation, returning a default value if it fails -/
def tryWithDefault {α : Type} (m : DiffExceptT α) (default : α) : Monadic.DiffM α := do
  let result ← runDiffExcept m
  match result with
  | Except.ok a => pure a
  | Except.error _ => pure default

/-- Check if a float value is finite (not NaN or infinity) -/
def isFinite (x : Float) : Bool :=
  !(x.isNaN || x.isInf)

/-- Check if a Value Float has finite values -/
def hasFiniteValues (v : Value Float) : Bool :=
  isFinite v.primal && isFinite v.tangent

/-- Run a validation check on a differentiable computation -/
def validateComputation {α : Type} (m : Monadic.DiffM α) (validator : α → Bool) (errorMsg : String) : DiffExceptT α := do
  let result ← ExceptT.lift m
  if validator result then
    pure result
  else
    throw (DiffError.other errorMsg)

end ExceptTransformer

end LeanDidax2
