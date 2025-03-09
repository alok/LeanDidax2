/-
  StateT Adapter for LeanDidax.
  This module provides adapters for using Lean's built-in StateT monad transformer
  with differentiable computations.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.MonadTransformer
import LeanDidax2.Utils

namespace LeanDidax2

namespace StateTransformer

/--
  The DiffStateT type is an abbreviation for StateT with Monadic.DiffM as the inner monad.
  This allows tracking and updating state throughout a differentiable computation.
-/
abbrev DiffStateT (σ : Type) (α : Type) := StateT σ Monadic.DiffM α

/-- Run a DiffStateT computation with an initial state and get both result and final state -/
def runDiffState {σ α : Type} (initial : σ) (m : DiffStateT σ α) : Monadic.DiffM (α × σ) :=
  StateT.run m initial

/-- Run a DiffStateT computation with an initial state and return only the result -/
def evalDiffState {σ α : Type} (initial : σ) (m : DiffStateT σ α) : Monadic.DiffM α := do
  let (a, _) ← runDiffState initial m
  pure a

/-- Run a DiffStateT computation with an initial state and return only the final state -/
def execDiffState {σ α : Type} (initial : σ) (m : DiffStateT σ α) : Monadic.DiffM σ := do
  let (_, s) ← runDiffState initial m
  pure s

/-- Apply a function to a value, tracking derivatives, and update state -/
def applyWithState {σ α β : Type} (f : α → Monadic.DiffM β) (x : α) (updateFn : α → β → σ → σ)
                  : DiffStateT σ β := do
  let result ← StateT.lift (f x)
  let s ← get
  set (updateFn x result s)
  pure result

/-- Run a differentiable computation with state tracking -/
def withDiffState {σ α : Type} (initial : σ) (m : DiffStateT σ α) : Monadic.DiffM (α × σ) :=
  runDiffState initial m

/--
  Execute a DiffM computation and pass the result to an IO action.
  This bridges the gap between differentiable computations and IO.
-/
def runDiffStateIO {σ α β : Type} (initial : σ) (m : DiffStateT σ α) (f : α → σ → IO β) : IO β := do
  -- Run the state computation and get the result pair
  let diffResult := runDiffState initial m
  -- Extract the actual values from the DiffM result
  let diffValues := Monadic.runDiff diffResult
  -- We know the structure is Value (α × σ), so extract the components
  let a := (diffValues.primal).1
  let s := (diffValues.primal).2
  -- Pass the extracted values to the IO function
  f a s

/--
  ADState allows tracking computation metadata during differentiation.
  This can be used to collect statistics or debug information.
-/
structure ADState where
  /-- Count of mathematical operations -/
  opCount : Nat := 0
  /-- Count of variable references -/
  varCount : Nat := 0
  /-- Maximum tangent value encountered -/
  maxTangent : Float := 0.0
  /-- History of operations performed -/
  opHistory : Array String := #[]
  /-- Flag indicating if tracing is enabled -/
  traceEnabled : Bool := false

/-- Helper to track operation in the AD state -/
def trackOp (opName : String) : DiffStateT ADState Unit := do
  let state ← get
  let newState := {
    state with
    opCount := state.opCount + 1,
    opHistory := if state.traceEnabled then
                   state.opHistory.push opName
                 else
                   state.opHistory
  }
  set newState

/-- Helper to track tangent value in the AD state -/
def trackTangent (v : Value Float) : DiffStateT ADState Unit := do
  let state ← get
  let absTangent := Float.abs v.tangent
  let newMaxTangent := if absTangent > state.maxTangent then absTangent else state.maxTangent
  set { state with maxTangent := newMaxTangent }

/-- Perform a differentiable operation with tracking -/
def trackedOp (opName : String) (x : Value Float) (y : Value Float)
              (f : Value Float → Value Float → Value Float) : DiffStateT ADState (Value Float) := do
  trackOp opName
  let result := f x y
  trackTangent result
  pure result

/-- Reset the AD state -/
def resetState : DiffStateT ADState Unit := do
  set (ADState.mk 0 0 0.0 #[] false)

/-- Enable or disable tracing in the AD state -/
def setTracing (enabled : Bool) : DiffStateT ADState Unit := do
  let state ← get
  set { state with traceEnabled := enabled }

/-- Get a summary of the operation history -/
def getOpSummary : DiffStateT ADState String := do
  let state ← get
  pure s!"Operations: {state.opCount}, Variables: {state.varCount}, Max Tangent: {state.maxTangent}"

/-- Create a constant Value -/
def constValue (x : Float) : Value Float :=
  { primal := x, tangent := 0.0 }

end StateTransformer

end LeanDidax2
