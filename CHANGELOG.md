# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2024-06-22

### Added
- WriterTransformer implementation for logging differentiable computations
- LoggedOperation type for tracking results with associated logs
- Binary operation logging with input/output tracking
- Utilities for filtering and formatting logs
- Timestamp formatting for sequential operation tracing
- Examples demonstrating logger usage with polynomial functions

### Improved
- Enhanced debugging capabilities with structured logging
- Better visualization of computation steps
- More examples of tracking differentiable operations

## [0.7.0] - 2024-06-21

### Added
- ExceptTransformer implementation for error handling in differentiable computations
- DiffError type with domain, arithmetic, and numerical error variants
- Safe mathematical operations (safelog, safediv, safesqrt) that handle domain errors
- Error handling utilities including tryWithDefault and catchDomain
- Integration of state and error handling with DiffExceptStateT
- Comprehensive examples demonstrating error handling in differentiable computations

### Improved
- Enhanced domain validation for mathematical operations
- Better debugging of numerical instabilities in computations
- More comprehensive error reporting with structured error types
- Safer composable transformations that handle edge cases gracefully

## [0.6.0] - 2024-06-20

### Added
- StateTransformer implementation using Lean's built-in StateT monad transformer
- ADState tracking for detailed operation monitoring
- Bridge between differentiable computations and IO
- Helper functions for traceable differentiable operations
- Operation history collection for debugging

### Improved
- Integration with standard library monad transformers
- Better separation of differentiable computation from IO
- More comprehensive examples of stateful computation
- Enhanced debugging capabilities

## [0.5.0] - 2024-06-19

### Added
- Integration with Lean's standard library MonadLift typeclass
- More powerful transformer composition utilities
- Additional transformation metadata tracking
- Configurable debugging for transformations

### Improved
- Refactored monad lifting to use standard library typeclasses
- Simplified API for lifting functions between monads
- Enhanced ADConfig with more debugging options
- Better composition of transformations with metadata
- Improved documentation and examples

## [0.4.0] - 2024-06-18

### Added
- Monad transformer implementation for composable transformations
- ReaderT monad transformer for configuration-based autodiff
- JAX-like function transformation composition utilities
- Batch gradient application over arrays of inputs
- Utils module with additional type instances and helper functions
- Debugging utilities for transformations
- Examples demonstrating monad transformer usage

### Improved
- Enhanced documentation on using monad transformers
- Better integration between monadic interface and transformations
- More JAX-like transformation interface
- Fixed type compatibility issues between DiffM and DiffReaderT
- Added better ToString instances for Value types

## [0.3.0] - 2024-06-17

### Added
- Monadic interface for automatic differentiation with the DiffM monad
- Support for do-notation in differentiable computations
- Higher-order function composition with the monadic API

### Improved
- Performance optimizations by replacing List with Array in key functions
- Enhanced type class instances for numeric literals
- More intuitive operator usage through HMul, HAdd instances

## [0.2.0] - 2024-06-16

### Added
- Vectorized batching functionality with `vmap` transformation
- Custom derivative rules for defining function derivatives
- Control flow primitives for differentiable programming, including `cond`, `select`, and `switch`
- Examples for each new feature and extended documentation

### Improved
- Fixed linter errors in the codebase
- Improved type class instances
- Enhanced test coverage

## [0.1.0] - 2024-06-15

### Added
- Forward-mode automatic differentiation
- Reverse-mode automatic differentiation
- Basic mathematical operations (+, -, *, /)
- Trigonometric functions (sin, cos, tan)
- Hyperbolic functions (sinh, cosh, tanh)
- Exponential and logarithmic functions (exp, log)
- Activation functions (sigmoid, relu)
- Example demonstrating usage

## [0.0.1] - 2024-06-14

### Added
- Initial project structure for LeanDidax automatic differentiation system
- Basic Value type to track primal and tangent values
- Forward-mode differentiation of simple expressions
- Reverse-mode gradient using computational graph

### Fixed
- Corrected derivative calculation for composite functions 