/-
  Monad Transformer implementation for LeanDidax.
  This module implements monad transformers to compose autodiff capabilities
  with other effects, similar to JAX's composable transformation system.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.Batch

namespace LeanDidax2

/-
  In JAX, transformations like grad, jit, vmap can be composed:
  jax.jit(jax.grad(jax.vmap(f)))

  This module implements a similar concept using monad transformers,
  allowing the composition of different autodiff capabilities.
-/
namespace MonadTransformer

/--
  ReaderT transformer for configuration parameters.
  This is a standard monad transformer that adds a configuration environment
  to an underlying monad.
-/
structure ReaderT (ρ : Type) (M : Type → Type) [Monad M] (α : Type) where
  /-- The function that executes the reader computation -/
  run : ρ → M α
  deriving Inhabited

/-- Monad instance for ReaderT -/
instance [Monad M] {ρ : Type} : Monad (ReaderT ρ M) where
  pure {α} (a : α) := { run := fun _ => pure a }

  bind {α β} (ma : ReaderT ρ M α) (f : α → ReaderT ρ M β) := {
    run := fun r => do
      let a ← ma.run r
      (f a).run r
  }

  map {α β} (f : α → β) (ma : ReaderT ρ M α) := {
    run := fun r => do
      let a ← ma.run r
      pure (f a)
  }

/-- Configuration for autodiff transformations -/
structure ADConfig where
  /-- Whether to use forward mode -/
  forwardMode : Bool := true
  /-- Initial seed value for gradients -/
  seed : Float := 1.0
  /-- Whether to retain the computation graph -/
  retainGraph : Bool := false
  /-- Epsilon value for numeric differentiation -/
  epsilon : Float := 1e-10
  /-- Whether to enable debugging mode -/
  debug : Bool := false

/--
  The DiffReaderT monad transformer adds a configuration
  environment to differentiable computations.
-/
abbrev DiffReaderT (ρ : Type) (α : Type) := ReaderT ρ Monadic.DiffM α

/-- Create a DiffReaderT computation with a specific configuration -/
def withConfig {ρ α : Type} (config : ρ) (m : DiffReaderT ρ α) : Monadic.DiffM α :=
  m.run config

/-- Get the current configuration in a DiffReaderT computation -/
def ask {ρ : Type} : DiffReaderT ρ ρ := {
  run := fun r => { run := fun _ => { primal := r, tangent := r } }
}

/-- Modify the configuration for a computation -/
def localConfig {ρ α : Type} (f : ρ → ρ) (m : DiffReaderT ρ α) : DiffReaderT ρ α := {
  run := fun r => m.run (f r)
}

/--
  Apply a function to the current environment and use the result in
  a computation.
-/
def withReader {ρ ρ' α : Type} (f : ρ → ρ') (x : DiffReaderT ρ' α) : DiffReaderT ρ α := {
  run := fun r => x.run (f r)
}

/-- Perform a computation with a modified environment -/
def withModifiedConfig {ρ α : Type} (f : ρ → ρ) (x : DiffReaderT ρ α) : DiffReaderT ρ α := do
  let oldEnv ← ask
  let result ← localConfig f x
  pure result

/--
  TransformedFunction represents a function with a set of transformations applied.
  This is similar to how JAX allows transformations to be composed and tracked.
-/
structure TransformedFunction (α : Type) (β : Type) where
  /-- The transformed function -/
  apply : α → β
  /-- The transformations that were applied, for introspection -/
  transformations : List String
  /-- Optional metadata about the transformation -/
  metadata : Option (Array String) := none

/-- Apply the grad transformation to a function -/
def gradTransform (f : Monadic.DiffM Float → Monadic.DiffM Float) : TransformedFunction Float Float :=
  {
    apply := Monadic.grad f,
    transformations := ["grad"]
  }

/-- Apply the vmap transformation to a function -/
def vmapTransform {α β : Type} (f : α → β) : TransformedFunction (Array α) (Array β) :=
  {
    apply := fun xs => xs.map f,
    transformations := ["vmap"]
  }

/-- Compose two transformations -/
def composeTransforms {α β γ : Type}
  (f : TransformedFunction β γ)
  (g : TransformedFunction α β) : TransformedFunction α γ :=
  {
    apply := fun x => f.apply (g.apply x),
    transformations := g.transformations ++ f.transformations,
    metadata := match f.metadata, g.metadata with
      | some m1, some m2 => some (m2 ++ m1)
      | some m, none => some m
      | none, some m => some m
      | none, none => none
  }

/--
  Convenience function to apply the grad transformation to a batch of inputs.
  Similar to JAX's vmap(grad(f)).
-/
def batchGrad (f : Monadic.DiffM Float → Monadic.DiffM Float)
              (xs : Array Float) : Array Float :=
  xs.map (Monadic.grad f)

/--
  Convenience function to compute both values and gradients for a batch of inputs.
  Similar to JAX's vmap(value_and_grad(f)).
-/
def batchValueAndGrad (f : Monadic.DiffM Float → Monadic.DiffM Float)
                     (xs : Array Float) : Array (Float × Float) :=
  xs.map (Monadic.valueAndGrad f)

/--
  Higher-order function to configure a computation with a specific config
  and then run it.
-/
def withADConfig {α : Type} (config : ADConfig)
                (computation : DiffReaderT ADConfig α) : Monadic.DiffM α :=
  withConfig config computation

/-- Apply a transformation in a configurable environment -/
def configuredTransform {α β : Type} [ToString α] [ToString β]
    (name : String)
    (f : α → β)
    (config : ADConfig) : TransformedFunction α β :=
  {
    apply := fun x =>
      let result := f x
      if config.debug then
        dbg_trace s!"[{name}] {x} => {result}"
        result
      else
        result,
    transformations := [name],
    metadata := some #[s!"debug: {config.debug}"]
  }

/-- Create a monad transformer function with configuration -/
def transformWithConfig {α β : Type} [ToString α]
    (transform : TransformedFunction α β)
    (config : ADConfig) : TransformedFunction α β :=
  {
    transform with
    apply := fun x =>
      if config.debug then
        dbg_trace s!"[{transform.transformations}] Apply to {x}"
        transform.apply x
      else
        transform.apply x
  }

end MonadTransformer

end LeanDidax2
