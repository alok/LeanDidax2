/-
  LeanDidax2: A pedagogical implementation of automatic differentiation in Lean 4.
  Inspired by JAX's Autodidax tutorial at https://docs.jax.dev/en/latest/autodidax.html
-/

-- This module serves as the root of the `LeanDidax2` library.
-- Import core modules here that should be built as part of the library.
import LeanDidax2.Basic
import LeanDidax2.Autodiff
import LeanDidax2.Batch
import LeanDidax2.CustomRules
import LeanDidax2.ControlFlow
import LeanDidax2.Monadic
import LeanDidax2.MonadTransformer
import LeanDidax2.StateTransformer
import LeanDidax2.ExceptTransformer
import LeanDidax2.WriterTransformer
import LeanDidax2.Utils
import LeanDidax2.Jaxpr
import LeanDidax2.Jit
import LeanDidax2.JitExamples
-- Example modules are not imported here to avoid main function conflicts.
-- They can be run directly using:
--   lake env lean --run LeanDidax2/Examples.lean
--   lake env lean --run LeanDidax2/BatchExamples.lean
--   lake env lean --run LeanDidax2/CustomRulesExamples.lean
--   lake env lean --run LeanDidax2/ControlFlowExamples.lean
--   lake env lean --run LeanDidax2/MonadicExamples.lean
--   lake env lean --run LeanDidax2/MonadTransformerExamples.lean
--   lake env lean --run LeanDidax2/StateTransformerExamples.lean
--   lake env lean --run LeanDidax2/ExceptTransformerExamples.lean
--   lake env lean --run LeanDidax2/WriterTransformerExamples.lean
--   lake env lean --run LeanDidax2/JitExamples.lean

/-
  LeanDidax2 implements an automatic differentiation system in Lean 4,
  following the design principles of JAX's Autodidax tutorial.

  Main components:
  - Forward-mode autodiff (Basic.lean)
  - Reverse-mode autodiff (Autodiff.lean)
  - Vectorized operations (Batch.lean)
  - Custom derivative rules (CustomRules.lean)
  - Control flow primitives (ControlFlow.lean)
  - Monadic implementation (Monadic.lean)
  - Monad transformers for composable transformations (MonadTransformer.lean)
  - State transformers for stateful computations (StateTransformer.lean)
  - Exception handling for error management (ExceptTransformer.lean)
  - Logging utilities for operation tracing (WriterTransformer.lean)
  - Utility functions and instances (Utils.lean)
  - Jaxpr intermediate representation (Jaxpr.lean)
  - JIT compilation functionality (Jit.lean)

  Each component includes example files demonstrating its usage.
-/

namespace LeanDidax2

/-!
  # LeanDidax2

  This library provides a Lean 4 implementation of JAX-style automatic differentiation.
  It includes both forward-mode and reverse-mode differentiation, as well as support
  for batching, control flow, and JIT compilation.

  ## Features

  * Forward-mode autodiff with dual numbers
  * Reverse-mode autodiff using computational graphs
  * Batch processing via the `vmap` transformation
  * Differentiable control flow primitives (`cond`, `switch`, and `select`)
  * Custom derivative rules for user-defined functions
  * JIT compilation for staged computation (inspired by JAX's Part 3)

  ## Main modules:

  * `Basic`: Core types and operations
  * `Autodiff`: Forward-mode and reverse-mode differentiation
  * `Batch`: Vectorized operations with `vmap`
  * `ControlFlow`: Differentiable control flow primitives
  * `CustomRules`: Define custom derivative rules
  * `Jaxpr`: Intermediate representation for JIT and transforms
  * `Jit`: Just-in-time compilation
-/

/-- LeanDidax2 version number. -/
def version : String := "0.3.0"

end LeanDidax2
