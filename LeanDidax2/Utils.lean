/-
  Utility functions and instances for the LeanDidax library.
  Provides extra type instances and conversion functions.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.MonadTransformer

namespace LeanDidax2

namespace Utils

/-- ToString instance for Value Float -/
instance : ToString (Value Float) where
  toString (v : Value Float) := s!"Value({v.primal}, {v.tangent})"

/-- ToString instance for Value type with a generic parameter α -/
instance [ToString α] : ToString (Value α) where
  toString (v : Value α) := s!"Value({v.primal}, {v.tangent})"

/--
  MonadLift instance to lift a DiffM computation into a DiffReaderT computation.
  This allows DiffM operations to be used within a DiffReaderT context.
-/
instance {ρ : Type} : MonadLift Monadic.DiffM (MonadTransformer.DiffReaderT ρ) where
  monadLift {α} (m : Monadic.DiffM α) : MonadTransformer.DiffReaderT ρ α := {
    run := fun _ => m
  }

/-- Legacy function for backward compatibility -/
def liftDiffM {ρ α : Type} (m : Monadic.DiffM α) : MonadTransformer.DiffReaderT ρ α :=
  monadLift m

/-- Run a computation with local modification to the environment -/
def locally {ρ α : Type} (f : ρ → ρ) (r : ρ) (m : MonadTransformer.DiffReaderT ρ α) : Monadic.DiffM α :=
  (MonadTransformer.localConfig f m).run r

/--
  Apply a monadic function to a value with an environment.
  This is similar to a natural transformation, adapted for our specific monad types.
-/
def withEnv {ρ α β : Type}
    (f : Monadic.DiffM α → Monadic.DiffM β)
    (x : MonadTransformer.DiffReaderT ρ α) : MonadTransformer.DiffReaderT ρ β := {
  run := fun r => f (x.run r)
}

/--
  Chain monadic functions with automatic lifting.
  This provides a more convenient way to compose functions than manual binding.
-/
def chainFunctions {α β γ : Type}
    (f : α → Monadic.DiffM β)
    (g : β → Monadic.DiffM γ)
    (x : α) : Monadic.DiffM γ := do
  let y ← f x
  g y

/-- Create a function transformer that prints debugging information -/
def debugTransform {α β : Type} [ToString α] [ToString β]
    (name : String)
    (f : α → β) : α → β :=
  fun x =>
    let result := f x
    dbg_trace s!"[{name}] {x} => {result}"
    result

/--
  Higher-order function to transform a function and automatically lift it
  to work with DiffReaderT. This provides a cleaner interface for composing
  transformations with configuration.
-/
def liftTransform {ρ α β : Type}
    (f : α → Monadic.DiffM β)
    (x : α) : MonadTransformer.DiffReaderT ρ β :=
  monadLift (f x)

/-- Function to compose transformed functions with better type handling -/
def composeTransformsBetter {α β γ : Type}
  (f : MonadTransformer.TransformedFunction β γ)
  (g : MonadTransformer.TransformedFunction α β) : MonadTransformer.TransformedFunction α γ :=
  {
    apply := fun x => f.apply (g.apply x),
    transformations := g.transformations ++ f.transformations
  }

/-- Map a transformation over each element of an array -/
def mapArrayTransform {α β : Type}
  (f : MonadTransformer.TransformedFunction α β)
  : MonadTransformer.TransformedFunction (Array α) (Array β) :=
  {
    apply := fun xs => xs.map f.apply,
    transformations := ["map"] ++ f.transformations
  }

/-- Helper function to create a DiffM Float constant -/
def diffMConstant (x : Float) : Monadic.DiffM Float :=
  { run := fun _ => { primal := x, tangent := 0 } }

/-- Convert numeric literals directly to DiffM Float values -/
instance : Coe Float (Monadic.DiffM Float) := ⟨diffMConstant⟩

/-- Convert numeric literals directly to DiffReaderT values -/
instance {ρ : Type} : Coe Float (MonadTransformer.DiffReaderT ρ Float) :=
  ⟨fun x => monadLift (diffMConstant x)⟩

/-- Default instance for TransformedFunction Float Float -/
instance : Inhabited (MonadTransformer.TransformedFunction Float Float) where
  default := {
    apply := fun x => x,
    transformations := ["identity"]
  }

/-- Make an array of transformed float-to-float functions into one that processes arrays -/
def batchTransform (fs : Array (MonadTransformer.TransformedFunction Float Float))
  : MonadTransformer.TransformedFunction (Array Float) (Array Float) :=
  {
    apply := fun xs =>
      if xs.size ≤ fs.size then
        Id.run do
          let mut result := xs
          for i in [:xs.size] do
            if i < fs.size then
              result := result.set! i (fs[i]!.apply xs[i]!)
          result
      else
        xs.map (fun x => (fs[0]!).apply x),
    transformations := ["batch"] ++ fs.foldl (fun acc f => acc ++ f.transformations) []
  }

end Utils

end LeanDidax2
