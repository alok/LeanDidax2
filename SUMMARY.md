# LeanDidax2: A JAX-inspired Automatic Differentiation System in Lean 4

## Project Overview

LeanDidax2 is a pedagogical implementation of automatic differentiation inspired by the [JAX Autodidax tutorial](https://docs.jax.dev/en/latest/autodidax.html). It demonstrates both forward-mode and reverse-mode automatic differentiation in a pure functional programming language (Lean 4).

## Key Features

### Core Components

1. **Forward-Mode Autodiff** (`Basic.lean`)
   - Dual number implementation with Value type (primal + tangent)
   - Operator overloading via typeclasses
   - Elementary functions (sin, cos, exp, log, etc.)
   - Higher-order differentiation functions

2. **Reverse-Mode Autodiff** (`Autodiff.lean`)
   - Computational graph representation
   - Backward pass for efficient gradient computation
   - Support for multi-input/multi-output functions

3. **Vectorized Operations** (`Batch.lean`)
   - JAX-like `vmap` transformation for batched operations
   - Efficient array-based implementation
   - Jacobian computation for vector-valued functions

4. **Control Flow Primitives** (`ControlFlow.lean`)
   - Differentiable conditionals (`cond`, `select`, `switch`)
   - Support for branching in differentiable programs
   - Piecewise function composition

5. **Custom Derivative Rules** (`CustomRules.lean`)
   - Custom rule registry for specialized derivatives
   - Extensible system for adding new functions
   - Demonstration of rule composition

6. **Monadic Interface** (`Monadic.lean`)
   - Monadic approach using `DiffM` monad
   - Support for do-notation in differentiable computations
   - Higher-order function composition
   - Demonstration of functional programming idioms

7. **Monad Transformers** (`MonadTransformer.lean`)
   - JAX-like composable transformations
   - ReaderT monad transformer for configuration
   - Integration with Lean's standard MonadLift typeclass
   - Transformation metadata and tracking
   - Function composition utilities
   - Configurable debugging for transformations

8. **State Transformers** (`StateTransformer.lean`)
   - StateT integration for stateful differentiable computations
   - Operation tracking and history collection
   - Performance monitoring and statistics gathering
   - Bridging between differentiable computations and IO
   - Debugging utilities with rich state information

9. **Error Handling** (`ExceptTransformer.lean`)
   - ExceptT monad transformer for domain error handling
   - Safe mathematical operations with explicit error handling
   - DiffError type hierarchy for structured error reporting
   - Integration with state transformers for combined effects
   - Error recovery strategies and fallback mechanisms
   - Domain validation for numerical stability

10. **Logging** (`WriterTransformer.lean`)
   - LoggedOperation type for tracking results with associated logs
   - Operation-level logging of inputs and outputs
   - Detailed trace visualization for computation pipelines
   - Filtering and formatting utilities for logs
   - Timestamp simulation for sequential operation tracing
   - Examples of integrating logging with differentiable computations

11. **Utility Functions** (`Utils.lean`)
   - Additional type instances and conversions
   - Debugging utilities for transformations
   - Enhanced integration between monads and transformers
   - Standard MonadLift instances for cross-monad operations
   - Higher-order functions for transformer composition

### Implementation Highlights

- **Pure Functional Programming**: Implemented in Lean 4 with focus on functional composition
- **Typeclass Integration**: Extensive use of Lean's typeclass system for operator overloading
- **Standard Library Integration**: Leveraging Lean's monad transformer infrastructure
- **Performance Optimization**: Array-based implementation for improved performance
- **Modular Design**: Each component is isolated in its own module with clear interfaces
- **Extensive Examples**: Each feature is accompanied by examples demonstrating its usage
- **Monad Transformers**: Use of monad transformer stack for composable effects
- **Stateful Computation**: Supporting stateful operations while maintaining functional purity
- **Error Handling**: Comprehensive error handling with domain validation and recovery

## Project Structure

```
LeanDidax2/
├── Basic.lean             # Forward-mode autodiff core
├── Autodiff.lean          # Reverse-mode autodiff implementation
├── Batch.lean             # Vectorized operations implementation
├── BatchExamples.lean     # Examples of batched operations
├── ControlFlow.lean       # Differentiable control flow primitives
├── ControlFlowExamples.lean # Examples of control flow usage
├── CustomRules.lean       # Custom derivative rules implementation
├── CustomRulesExamples.lean # Examples of custom rule usage
├── Monadic.lean           # Monadic interface implementation
├── MonadicExamples.lean   # Examples of monadic usage
├── MonadTransformer.lean  # Monad transformer implementation
├── MonadTransformerExamples.lean # Examples of transformer usage
├── StateTransformer.lean  # State transformer implementation
├── StateTransformerExamples.lean # Examples of stateful computation
├── ExceptTransformer.lean # Exception handling for differentiable computation
├── ExceptTransformerExamples.lean # Examples of error handling
├── WriterTransformer.lean # Logging utilities for computation tracing
├── WriterTransformerExamples.lean # Examples of logging and tracing
├── Utils.lean             # Utility functions and instances
└── Examples.lean          # General examples of autodiff
```

## Functional Programming Concepts Demonstrated

1. **Functors and Monads**: Used for chaining differentiable computations
2. **Monad Transformers**: Stacking monadic effects for composable transformations
3. **MonadLift**: Type-safe lifting of computations between monads
4. **StateT Monad**: Managing state in pure functional computations
5. **ExceptT Monad**: Error handling in pure functional computations
6. **Typeclasses**: Extensively used for operator overloading and type constraints
7. **Higher-Order Functions**: Functions that take functions as arguments
8. **Pure Functions**: No side effects in the differentiation system
9. **Composition**: Building complex operations from simpler ones
10. **Immutable Data**: All operations create new values rather than modifying existing ones

## Lean 4 Specific Features

1. **Do-Notation**: Used in monadic computations for readability
2. **Array Manipulation**: Efficient array operations for performance
3. **Type Inference**: Leveraging Lean's powerful type system
4. **Instance Resolution**: Automatic derivation of instances
5. **Unicode Support**: Using mathematical symbols where appropriate
6. **Typeclass Inheritance**: Building a hierarchy of mathematical operations
7. **MonadLift**: Standard library integration for transformers
8. **Built-in Monad Transformers**: Using StateT and ReaderT from the standard library

## JAX Concepts Implemented

1. **Automatic Differentiation**: Core JAX concept
2. **Batching Transformation**: Similar to JAX's vmap
3. **Control Flow**: Differentiable conditionals as in JAX's lax module
4. **Custom Derivatives**: Similar to JAX's custom_jvp
5. **Compositionality**: Building complex transforms from simple ones
6. **Function Transformations**: JAX-like function transformation with composition
7. **Transformation Metadata**: Similar to JAX's transformation tracing
8. **Stateful Computations**: Supporting stateful tracking and debugging

## Usage Example

```lean
import LeanDidax2.Basic

open LeanDidax2

-- Define a function to differentiate
def f (x : Value Float) : Value Float :=
  x * x + 2 * x + 1

-- Compute the derivative at x=3
#eval
  let x := seed 3.0
  let result := f x
  (s!"f(3) = {result.primal}", s!"f'(3) = {result.tangent}")
```

## Monadic Usage Example

```lean
import LeanDidax2.Monadic

open LeanDidax2.Monadic

-- Define a function using monadic style
def polynomialM (x : DiffM Float) : DiffM Float := do
  let x_squared := x * x
  let two_x := 2.0 * x
  let sum := x_squared + two_x
  sum + 1.0

-- Compute the gradient
#eval
  let x := 3.0
  let grad := grad polynomialM x
  s!"The gradient of x^2 + 2x + 1 at x=3 is {grad}"
```

## Monad Transformer Example

```lean
import LeanDidax2.MonadTransformer
import LeanDidax2.Utils

open LeanDidax2.MonadTransformer
open LeanDidax2.Utils

-- Define a function that uses configuration
def configuredFunction : DiffReaderT ADConfig Float := do
  let config ← ask
  let scale := if config.forwardMode then 1.0 else config.seed
  let x := monadLift (differentiableVar (2.0 * scale))
  liftPolynomial x

-- Run with different configurations
#eval
  let config1 : ADConfig := {}  -- default config
  let config2 : ADConfig := { forwardMode := false, seed := 3.0 }
  let result1 := runDiff (withADConfig config1 configuredFunction)
  let result2 := runDiff (withADConfig config2 configuredFunction)
  (s!"Result with default config: {result1}", 
   s!"Result with custom config: {result2}")
```

## State Transformer Example

```lean
import LeanDidax2.StateTransformer
import LeanDidax2.Utils

open LeanDidax2.StateTransformer
open LeanDidax2.Utils

-- Define a stateful computation
def polynomialWithState (x : Float) : DiffStateT ADState (Value Float) := do
  setTracing true
  let xv := seed x
  trackOp "seed"
  
  let x_squared ← trackedOp "multiply" xv xv (fun a b => a * b)
  let two_x ← trackedOp "multiply" (constValue 2.0) xv (fun a b => a * b)
  let result ← trackedOp "add" x_squared two_x (fun a b => a + b)
  
  pure result

-- Run the computation and get both result and operation history
#eval
  let initialState : ADState := {}
  runDiffStateIO initialState (polynomialWithState 3.0) (fun result state => do
    IO.println s!"Result: {result}"
    IO.println s!"Operations tracked: {state.opCount}"
    IO.println s!"Max tangent: {state.maxTangent}"
  )
```

## Error Handling Example

```lean
import LeanDidax2.ExceptTransformer
import LeanDidax2.StateTransformer

open LeanDidax2.ExceptTransformer
open LeanDidax2.StateTransformer

-- Define a potentially unsafe computation
def riskyFunction (x : Value Float) : DiffExceptT (Value Float) := do
  -- This will fail for inputs that make the logarithm argument negative
  let result ← safelog (x * x - constValue 4.0)
  pure (result * constValue 2.0)

-- Example of handling errors with different strategies
#eval
  -- Run with proper error handling
  runDiffExceptIO
    (riskyFunction (seed 3.0))
    (fun result => IO.println s!"Success! Result = {result}")
    (fun error => IO.println s!"Failed: {error}")
    
  -- Run with a default value for errors
  let withDefault := tryWithDefault (riskyFunction (seed 1.0)) { primal := 0, tangent := 0 }
  IO.println s!"Result with default = {runDiff withDefault}"
```

## Logging Example

```lean
import LeanDidax2.WriterTransformer
import LeanDidax2.Basic

open LeanDidax2.WriterTransformer

-- Define a polynomial function with logging
def loggedPolynomial (x : Value Float) : Monadic.DiffM (LoggedOperation (Value Float)) := do
  -- Log each operation
  let x² ← loggedOp "square" x x (fun a b => a * b)
  let term2 ← loggedOp "2x" (constValue 2.0) x (fun a b => a * b)
  let sum ← loggedOp "x² + 2x" x².result term2.result (fun a b => a + b)
  let result ← loggedOp "sum + 1" sum.result (constValue 1.0) (fun a b => a + b)
  
  -- Combine all logs
  let allLogs := #["Computing f(x) = x² + 2x + 1"] ++
                 x².logs ++
                 term2.logs ++
                 sum.logs ++
                 result.logs
  
  pure {
    result := result.result,
    logs := allLogs
  }

-- Run the computation and analyze the logs
#eval
  let computation := loggedPolynomial (seed 3.0)
  runDiffWithLogsIO computation (fun result logs => do
    IO.println s!"Result: {result.primal}" 
    IO.println s!"Gradient: {result.tangent}"
    IO.println "Operation trace:"
    IO.println (formatLogs logs)
  )
```

## Future Improvements

1. **Performance Optimizations**: Further optimize array operations
2. **Extended Function Library**: Add more mathematical functions
3. **Hessian Computation**: Support for second-order derivatives
4. **GPU Acceleration**: Potential integration with Lean's FFI
5. **Neural Network Support**: Add specialized functions for ML applications
6. **More Transformations**: Additional JAX-like transformations (jit, pmap, etc.)
7. **Higher-kinded Typeclasses**: More advanced typeclass abstractions
8. **Profiling Integration**: Runtime performance monitoring
9. **Automatic Checkpointing**: Memory optimization for large computations
10. **Extended Error Handling**: More sophisticated error recovery mechanisms
11. **Numerical Stability**: Advanced techniques for maintaining computational stability

## Conclusion

LeanDidax2 demonstrates how automatic differentiation can be implemented in a pure functional language like Lean 4. By following the design principles of JAX, it provides an educational resource for understanding how modern autodiff systems work, while showcasing the expressive power of functional programming and dependent types. The addition of monad transformers, state transformers, exception handling, and integration with Lean's standard library further illustrates how complex transformations can be composed in a modular and type-safe manner. 