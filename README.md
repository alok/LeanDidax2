(This is 100% vibe coded by AI). i did it to show the PVS dev the current sota.

# LeanDidax

A pedagogical implementation of automatic differentiation in Lean 4, inspired by JAX's Autodidax tutorial.

For a comprehensive overview of the project and its components, see the [Summary Document](SUMMARY.md).

## Overview

LeanDidax is an automatic differentiation library written in Lean 4. It provides both forward-mode and reverse-mode autodiff capabilities, allowing users to compute derivatives of complex mathematical functions efficiently.

The library is inspired by the [JAX Autodidax tutorial](https://docs.jax.dev/en/latest/autodidax.html), which explains how to build a simple automatic differentiation system from scratch.

## Features

- **Forward-mode autodiff**: Efficient computation of derivatives using dual numbers
- **Reverse-mode autodiff**: Efficient computation of gradients using computational graphs
- **Operator overloading**: Use standard mathematical notation with autodiff types
- **Vectorized operations**: Support for batched operations and the `vmap` transformation
- **Control flow primitives**: Differentiable conditionals with `cond`, `select`, and `switch`
- **Custom derivative rules**: Define specialized derivatives for functions
- **Rich function library**: Support for common mathematical functions:
  - Basic operations: +, -, *, /
  - Trigonometric functions: sin, cos, tan
  - Hyperbolic functions: sinh, cosh, tanh
  - Exponential and logarithmic functions: exp, log
  - Non-differentiable functions with special handling: abs, relu
  - Activation functions for neural networks: sigmoid
- **Higher-order differentiation functions**: grad, valueAndGrad, jvp

## Installation

Add LeanDidax to your Lean project by adding it as a dependency in your `lakefile.lean`:

```lean
require LeanDidax from git "https://github.com/yourusername/LeanDidax" @ "main"
```

## Usage

### Basic Example: Forward-Mode Autodiff

```lean
import LeanDidax2.Basic

open LeanDidax2

-- Define a function to differentiate
def f (x : Value Float) : Value Float :=
  x * x + 2 * x + 1

-- Compute the value and derivative at x=3
#eval 
  let x := seed 3.0  -- Create a variable with tangent=1.0
  let result := f x
  (s!"Function value at x=3: {result.primal}",
   s!"Derivative at x=3: {result.tangent}")
```

### Reverse-Mode Autodiff Example

```lean
import LeanDidax2.Autodiff

open LeanDidax2
open LeanDidax2.ReverseMode

-- Define a function using a computational graph
def polynomial (x : Node) : Node :=
  let x2 := Node.Mul x x            -- x^2
  let two := Node.Leaf 2.0
  let twoX := Node.Mul two x        -- 2*x
  let sum1 := Node.Add x2 twoX      -- x^2 + 2*x
  let one := Node.Leaf 1.0
  Node.Add sum1 one                 -- x^2 + 2*x + 1

-- Compute the gradient at x=3
#eval
  let x := 3.0
  let xNode := Node.Leaf x
  let polyGraph := polynomial xNode
  let gradResults := backward polyGraph 1.0
  let grad := match gradResults.find? (fun pair => pair.1 == x) with
              | some (_, cotangent) => cotangent
              | none => 0.0
  (s!"Gradient of f(x) = x^2 + 2x + 1 at x=3: {grad}")
```

### Vectorized Operations with vmap

```lean
import LeanDidax2.Batch

open LeanDidax2
open LeanDidax2.Batch

-- Define a function
def f (x : Value Float) : Value Float := 
  x * x + 2 * x + 1

-- Apply it to multiple inputs at once
#eval
  let inputs := #[1.0, 2.0, 3.0, 4.0, 5.0]
  let results := vmap f inputs
  for i in [:inputs.size] do
    IO.println s!"{inputs[i]!}: value = {results[i]!.primal}, derivative = {results[i]!.tangent}"
```

### Custom Derivative Rules

```lean
import LeanDidax2.CustomRules

open LeanDidax2
open LeanDidax2.CustomRules

-- Define a function with a custom derivative rule
def f (x : Float) : Float := x * x * x * Float.sin x
def df (x : Float) : Float := 3 * x * x * Float.sin x + x * x * x * Float.cos x

-- Use the custom rule
#eval
  let x := 2.0
  let result := defCustomFn "cubic_sine" f df (seed x)
  (s!"Value: {result.primal}", s!"Derivative: {result.tangent}")
```

### Differentiable Control Flow

```lean
import LeanDidax2.ControlFlow

open LeanDidax2
open LeanDidax2.ControlFlow

-- Use differentiable conditionals
#eval
  let x := seed 2.0
  let result := cond (x.primal > 0.0)
    (fun _ => x * x)         -- x^2 when x > 0
    (fun _ => x * -1.0)      -- -x when x ≤ 0
  (s!"Value: {result.primal}", s!"Derivative: {result.tangent}")
```

## Implementation Details

### Value Type

The core of LeanDidax is the `Value` type, which tracks both the primal value and its tangent (derivative) information:

```lean
structure Value (α : Type) where
  primal : α
  tangent : α := primal
```

### Forward-Mode Autodiff

Forward-mode autodiff works by propagating tangent values alongside primal values through each operation. For example, the multiplication rule:

```lean
def mul [Mul α] [Add α] (x y : Value α) : Value α :=
  { primal := x.primal * y.primal,
    tangent := x.tangent * y.primal + x.primal * y.tangent }
```

### Reverse-Mode Autodiff

Reverse-mode autodiff builds a computational graph and uses backward propagation to compute gradients:

```lean
inductive Node
  | Leaf (value : Float)
  | Add (left right : Node)
  | Mul (left right : Node)
  | ...
```

### Control Flow Primitives

The library supports differentiable control flow:

```lean
def cond {α : Type} [Zero α] 
  (pred : Bool) 
  (trueBranch : Unit → Value α) 
  (falseBranch : Unit → Value α) : Value α :=
  if pred then trueBranch () else falseBranch ()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
