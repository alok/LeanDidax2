/-
  Monadic implementation of automatic differentiation in LeanDidax.
  This module enhances the basic autodiff system with monadic operations.
-/

import LeanDidax2.Basic

namespace LeanDidax2

namespace Monadic

/-
  In this module, we implement a monadic approach to automatic differentiation.
  This allows us to:
  1. Compose differentiable computations more elegantly
  2. Handle context and state throughout the computation
  3. Support advanced features like higher-order differentiation
-/

/--
  The DiffM monad represents a differentiable computation that
  produces a value of type α along with its tangent.

  This is essentially a State monad where the state contains information
  about the differentiation context.
-/
structure DiffM (α : Type) where
  /-- The function that runs the differentiable computation -/
  run : Unit → Value α
  deriving Inhabited

/-- Monad instance for DiffM -/
instance : Monad DiffM where
  pure {α} (a : α) := { run := fun _ => { primal := a, tangent := a } }

  bind {α β} (ma : DiffM α) (f : α → DiffM β) := {
    run := fun s =>
      let va := ma.run s
      let mb := f va.primal
      let vb := mb.run s
      { primal := vb.primal,
        tangent := vb.tangent }
  }

  map {α β} (f : α → β) (ma : DiffM α) := {
    run := fun s =>
      let va := ma.run s
      { primal := f va.primal, tangent := f va.primal }
  }

/-- Convert a Value to a DiffM computation -/
def lift [Zero α] (v : Value α) : DiffM α := {
  run := fun _ => v
}

/-- Create a differentiable variable from a seed value -/
def differentiableVar (x : Float) : DiffM Float := {
  run := fun _ => seed x
}

/-- Run a differentiable computation and get the value and gradient -/
def runDiff {α} (m : DiffM α) : Value α :=
  m.run ()

/-- Addition in the DiffM monad -/
def add [Add α] (x y : DiffM α) : DiffM α := {
  run := fun s =>
    let vx := x.run s
    let vy := y.run s
    LeanDidax2.add vx vy
}

/-- Multiplication in the DiffM monad -/
def mul [Mul α] [Add α] (x y : DiffM α) : DiffM α := {
  run := fun s =>
    let vx := x.run s
    let vy := y.run s
    LeanDidax2.mul vx vy
}

/-- Negation in the DiffM monad -/
def neg [Neg α] (x : DiffM α) : DiffM α := {
  run := fun s =>
    let vx := x.run s
    LeanDidax2.neg vx
}

/-- Subtraction in the DiffM monad -/
def sub [Sub α] [HSub α α α] (x y : DiffM α) : DiffM α := {
  run := fun s =>
    let vx := x.run s
    let vy := y.run s
    LeanDidax2.sub vx vy
}

/-- Division in the DiffM monad -/
def div [Div α] [Mul α] [Add α] [Neg α] [HSub α α α] (x y : DiffM α) : DiffM α := {
  run := fun s =>
    let vx := x.run s
    let vy := y.run s
    LeanDidax2.div vx vy
}

/-- Power function in the DiffM monad -/
def pow (x : DiffM Float) (y : Float) : DiffM Float := {
  run := fun s =>
    let vx := x.run s
    LeanDidax2.pow vx y
}

/-- Sine function in the DiffM monad -/
def sin (x : DiffM Float) : DiffM Float := {
  run := fun s =>
    let vx := x.run s
    LeanDidax2.sin vx
}

/-- Cosine function in the DiffM monad -/
def cos (x : DiffM Float) : DiffM Float := {
  run := fun s =>
    let vx := x.run s
    LeanDidax2.cos vx
}

/-- Tangent function in the DiffM monad -/
def tan (x : DiffM Float) : DiffM Float := {
  run := fun s =>
    let vx := x.run s
    LeanDidax2.tan vx
}

/-- Exponential function in the DiffM monad -/
def exp (x : DiffM Float) : DiffM Float := {
  run := fun s =>
    let vx := x.run s
    LeanDidax2.exp vx
}

/-- Logarithm function in the DiffM monad -/
def log (x : DiffM Float) : DiffM Float := {
  run := fun s =>
    let vx := x.run s
    LeanDidax2.log vx
}

/--
  Gradient computation in the monadic style.

  This function computes the gradient of a monadic computation `f` with respect to its input
  at the point `x`.
-/
def grad (f : DiffM Float → DiffM Float) (x : Float) : Float :=
  let xVar := differentiableVar x
  let result := runDiff (f xVar)
  result.tangent

/--
  Value and gradient computation in the monadic style.

  This function computes both the value and gradient of a monadic computation `f`
  at the point `x`.
-/
def valueAndGrad (f : DiffM Float → DiffM Float) (x : Float) : (Float × Float) :=
  let xVar := differentiableVar x
  let result := runDiff (f xVar)
  (result.primal, result.tangent)

/--
  Operator overloading instances for DiffM
-/
instance [Add α] : Add (DiffM α) where
  add := add

instance [Mul α] [Add α] : Mul (DiffM α) where
  mul := mul

instance [Neg α] : Neg (DiffM α) where
  neg := neg

instance [Sub α] [HSub α α α] : Sub (DiffM α) where
  sub := sub

instance [Div α] [Mul α] [Add α] [Neg α] [HSub α α α] : Div (DiffM α) where
  div := div

/-- Conversion from Float to DiffM Float -/
instance : Coe Float (DiffM Float) where
  coe (x : Float) := {
    run := fun _ => { primal := x, tangent := 0 }
  }

/-- Allow natural number literals for DiffM Float -/
instance : OfNat (DiffM Float) n where
  ofNat := {
    run := fun _ => { primal := n.toFloat, tangent := 0 }
  }

/-- Multiplication between Float and DiffM Float -/
instance : HMul Float (DiffM Float) (DiffM Float) where
  hMul (x : Float) (y : DiffM Float) :=
    { run := fun s =>
        let vy := y.run s
        { primal := x * vy.primal, tangent := x * vy.tangent }
    }

/-- Addition between Float and DiffM Float -/
instance : HAdd Float (DiffM Float) (DiffM Float) where
  hAdd (x : Float) (y : DiffM Float) :=
    { run := fun s =>
        let vy := y.run s
        { primal := x + vy.primal, tangent := vy.tangent }
    }

/-- Subtraction between Float and DiffM Float -/
instance : HSub Float (DiffM Float) (DiffM Float) where
  hSub (x : Float) (y : DiffM Float) :=
    { run := fun s =>
        let vy := y.run s
        { primal := x - vy.primal, tangent := -vy.tangent }
    }

/-- Division between Float and DiffM Float -/
instance : HDiv Float (DiffM Float) (DiffM Float) where
  hDiv (x : Float) (y : DiffM Float) :=
    { run := fun s =>
        let vy := y.run s
        { primal := x / vy.primal, tangent := -x * vy.tangent / (vy.primal * vy.primal) }
    }

end Monadic

end LeanDidax2
