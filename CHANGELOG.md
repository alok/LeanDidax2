# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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