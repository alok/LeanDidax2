/-
  Basic definitions for the LeanDidax autodifferentiation library.
  This is a pedagogical implementation inspired by JAX's Autodidax tutorial.
-/

namespace LeanDidax2

/--
  Value represents a value with its gradient information.
  In a standard autodiff system, this would track both the primal value
  and the corresponding tangent/cotangent for differentiation.
-/
structure Value (α : Type) where
  /-- The primal value -/
  primal : α
  /-- The tangent value (for forward-mode) or cotangent value (for reverse mode) -/
  tangent : α := primal
  deriving Repr

/-- Instance of Inhabited for Value type, needed for arrays and other data structures -/
instance [Inhabited α] : Inhabited (Value α) where
  default := { primal := default, tangent := default }

/-- Create a constant value with no gradient information (zero tangent). -/
def const [Zero α] (x : α) : Value α := { primal := x, tangent := 0 }

/-- Create a seed variable with tangent=1.0 for differentiation w.r.t. this variable. -/
def seed (x : Float) : Value Float := { primal := x, tangent := 1.0 }

/-- Add two values, propagating gradient information. -/
def add [Add α] (x y : Value α) : Value α :=
  { primal := x.primal + y.primal,
    tangent := x.tangent + y.tangent }

/-- Multiply two values, propagating gradient information (forward-mode). -/
def mul [Mul α] [Add α] (x y : Value α) : Value α :=
  { primal := x.primal * y.primal,
    tangent := x.tangent * y.primal + x.primal * y.tangent }

/-- Negate a value, propagating gradient information. -/
def neg [Neg α] (x : Value α) : Value α :=
  { primal := -x.primal,
    tangent := -x.tangent }

/-- Subtract two values, propagating gradient information. -/
def sub [Sub α] [HSub α α α] (x y : Value α) : Value α :=
  { primal := x.primal - y.primal,
    tangent := x.tangent - y.tangent }

/-- Divide two values, propagating gradient information (forward-mode). -/
def div [Div α] [Mul α] [Add α] [Neg α] [HSub α α α] (x y : Value α) : Value α :=
  { primal := x.primal / y.primal,
    tangent := (x.tangent * y.primal - x.primal * y.tangent) / (y.primal * y.primal) }

/-- Power function (x^y) with gradient propagation. -/
def pow (x : Value Float) (y : Float) : Value Float :=
  let primalResult := Float.pow x.primal y
  { primal := primalResult,
    tangent := y * Float.pow x.primal (y - 1.0) * x.tangent }

/-- Square root function with gradient propagation. -/
def sqrt (x : Value Float) : Value Float :=
  let sqrtX := Float.sqrt x.primal
  { primal := sqrtX,
    tangent := 0.5 / sqrtX * x.tangent }

/-- Sine function with gradient propagation. -/
def sin (x : Value Float) : Value Float :=
  { primal := Float.sin x.primal,
    tangent := Float.cos x.primal * x.tangent }

/-- Cosine function with gradient propagation. -/
def cos (x : Value Float) : Value Float :=
  { primal := Float.cos x.primal,
    tangent := -Float.sin x.primal * x.tangent }

/-- Tangent function with gradient propagation. -/
def tan (x : Value Float) : Value Float :=
  let tanX := Float.tan x.primal
  { primal := tanX,
    tangent := (1.0 + tanX * tanX) * x.tangent }

/-- Hyperbolic sine function with gradient propagation. -/
def sinh (x : Value Float) : Value Float :=
  let sinhX := Float.sinh x.primal
  { primal := sinhX,
    tangent := Float.cosh x.primal * x.tangent }

/-- Hyperbolic cosine function with gradient propagation. -/
def cosh (x : Value Float) : Value Float :=
  let coshX := Float.cosh x.primal
  { primal := coshX,
    tangent := Float.sinh x.primal * x.tangent }

/-- Hyperbolic tangent function with gradient propagation. -/
def tanh (x : Value Float) : Value Float :=
  let tanhX := Float.tanh x.primal
  { primal := tanhX,
    tangent := (1.0 - tanhX * tanhX) * x.tangent }

/-- Natural logarithm with gradient propagation. -/
def log (x : Value Float) : Value Float :=
  { primal := Float.log x.primal,
    tangent := x.tangent / x.primal }

/-- Exponential function with gradient propagation. -/
def exp (x : Value Float) : Value Float :=
  let expX := Float.exp x.primal
  { primal := expX,
    tangent := expX * x.tangent }

/-- Absolute value function with gradient propagation. -/
def abs (x : Value Float) : Value Float :=
  let absX := Float.abs x.primal
  let derivative := if x.primal < 0.0 then -1.0 else if x.primal > 0.0 then 1.0 else 0.0
  { primal := absX,
    tangent := derivative * x.tangent }

/-- ReLU activation function (max(0,x)) with gradient propagation. -/
def relu (x : Value Float) : Value Float :=
  let reluX := if x.primal > 0.0 then x.primal else 0.0
  let derivative := if x.primal > 0.0 then 1.0 else 0.0
  { primal := reluX,
    tangent := derivative * x.tangent }

/-- Sigmoid activation function (1/(1+e^(-x))) with gradient propagation. -/
def sigmoid (x : Value Float) : Value Float :=
  let sigX := 1.0 / (1.0 + Float.exp (-x.primal))
  { primal := sigX,
    tangent := sigX * (1.0 - sigX) * x.tangent }

/-- Higher-order function to compute the gradient of a function at a point. -/
def grad (f : Value Float → Value Float) (x : Float) : Float :=
  let seedX := seed x
  let result := f seedX
  result.tangent

/-- Higher-order function to compute both value and gradient of a function at a point. -/
def valueAndGrad (f : Value Float → Value Float) (x : Float) : (Float × Float) :=
  let seedX := seed x
  let result := f seedX
  (result.primal, result.tangent)

/-- Higher-order function to compute the Jacobian-vector product of a function. -/
def jvp (f : Value Float → Value Float) (primals : Float) (tangents : Float) : (Float × Float) :=
  let x : Value Float := { primal := primals, tangent := tangents }
  let y := f x
  (y.primal, y.tangent)

----------------------------------
-- Typeclass Instances for Value
----------------------------------

/-- Instance to enable addition of Value with `+` operator -/
instance [Add α] : Add (Value α) where
  add := add

/-- Instance to enable multiplication of Value with `*` operator -/
instance [Mul α] [Add α] : Mul (Value α) where
  mul := mul

/-- Instance to enable negation of Value with unary `-` operator -/
instance [Neg α] : Neg (Value α) where
  neg := neg

/-- Instance to enable subtraction of Value with `-` operator -/
instance [Sub α] [HSub α α α] : Sub (Value α) where
  sub := sub

/-- Instance to enable division of Value with `/` operator -/
instance [Div α] [Mul α] [Add α] [Neg α] [HSub α α α] : Div (Value α) where
  div := div

/-- Enable implicit conversion from regular values to constants -/
instance [Zero α] [Add α] : Coe α (Value α) where
  coe := const

/-- Enable numeric literals with Value type -/
instance [Zero α] [OfNat α n] : OfNat (Value α) n where
  ofNat := const (OfNat.ofNat n)

end LeanDidax2
