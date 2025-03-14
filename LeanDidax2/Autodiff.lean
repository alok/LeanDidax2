/-
  Autodifferentiation functionality for LeanDidax.
  Implements forward and reverse mode automatic differentiation.
-/

import LeanDidax2.Basic

namespace LeanDidax2

/-
  ForwardMode implements forward-mode automatic differentiation.
  It tracks the primal value and its directional derivative.
-/
namespace ForwardMode

/--
  JVP (Jacobian-Vector Product) computation for a function.
  Given a function and an input with its tangent, computes the output and its tangent.
-/
def jvp {α β : Type} (f : Value α → Value β) (x : Value α) (dx : α) : Value β :=
  f { primal := x.primal, tangent := dx }

/--
  Numerical derivative approximation (for testing purposes).
  Uses finite differences to approximate the derivative.
-/
def numericGradient
  {α : Type} [Add α] [Sub α] [Mul α] [Div α] [HMul α Float α] [HDiv α Float α]
  [HMul Float α α] -- Float * α -> α
  (f : α → α) (x : α) (h : Float := 1e-5) : α :=
  let step := h * x
  (f (x + step) - f (x - step)) / (step + step)

end ForwardMode

/-
  ReverseMode implements reverse-mode automatic differentiation.
  More efficient for functions with many inputs and few outputs.
  This implementation works specifically with Float type.
-/
namespace ReverseMode

/-- A node in the computational graph for Float values -/
inductive Node
  | Leaf (value : Float)
  | Add (left right : Node)
  | Mul (left right : Node)
  | Neg (operand : Node)
  | Sub (left right : Node)
  | Div (left right : Node)
  | Sin (operand : Node)
  | Cos (operand : Node)
  | Log (operand : Node)
  | Exp (operand : Node)
  deriving Repr

/-- Evaluate a computational graph node to get its primal value -/
def eval (node : Node) : Float :=
  match node with
  | .Leaf value => value
  | .Add left right => eval left + eval right
  | .Mul left right => eval left * eval right
  | .Neg operand => -eval operand
  | .Sub left right => eval left - eval right
  | .Div left right => eval left / eval right
  | .Sin operand => Float.sin (eval operand)
  | .Cos operand => Float.cos (eval operand)
  | .Log operand => Float.log (eval operand)
  | .Exp operand => Float.exp (eval operand)

/-- Backward pass through the computational graph to compute gradients -/
def backward (node : Node) (cotangent : Float) : List (Float × Float) :=
  let rec traverse (node : Node) (cotangent : Float) : List (Float × Float) :=
    match node with
    | .Leaf value => [(value, cotangent)]
    | .Add left right =>
        traverse left cotangent ++ traverse right cotangent
    | .Mul left right =>
        let leftValue := eval left
        let rightValue := eval right
        traverse left (cotangent * rightValue) ++ traverse right (cotangent * leftValue)
    | .Neg operand =>
        traverse operand (-cotangent)
    | .Sub left right =>
        traverse left cotangent ++ traverse right (-cotangent)
    | .Div left right =>
        let leftValue := eval left
        let rightValue := eval right
        let leftCotangent := cotangent / rightValue
        let rightCotangent := -cotangent * leftValue / (rightValue * rightValue)
        traverse left leftCotangent ++ traverse right rightCotangent
    | .Sin operand =>
        let operandValue := eval operand
        let derivativeCotangent := cotangent * Float.cos operandValue
        traverse operand derivativeCotangent
    | .Cos operand =>
        let operandValue := eval operand
        let derivativeCotangent := cotangent * (-Float.sin operandValue)
        traverse operand derivativeCotangent
    | .Log operand =>
        let operandValue := eval operand
        let derivativeCotangent := cotangent / operandValue
        traverse operand derivativeCotangent
    | .Exp operand =>
        let operandValue := eval operand
        let expValue := Float.exp operandValue
        let derivativeCotangent := cotangent * expValue
        traverse operand derivativeCotangent
  traverse node cotangent

/--
  VJP (Vector-Jacobian Product) computation.
  Core of reverse-mode autodifferentiation.
-/
def vjp (f : Node → Node) (x : Node) (cotangent : Float) : List (Float × Float) :=
  backward (f x) cotangent

/--
  Compute the gradient of a function at a point.
  This is a convenience wrapper around vjp.
-/
def grad (f : Node → Node) (x : Float) : Float :=
  let leafNode := Node.Leaf x
  let result := vjp f leafNode 1.0
  -- The gradient is the cotangent of the input variable
  match result.find? (fun pair => pair.1 == x) with
  | some (_, cotangent) => cotangent
  | none => 0.0

end ReverseMode

/-
  Automatic differentiation functions for Array types.
  This would support array/matrix operations with automatic differentiation.
-/
namespace ArrayAutodiff
  -- This would implement array/matrix operations with autodiff
  -- For simplicity, we're leaving this as a stub
end ArrayAutodiff

end LeanDidax2
