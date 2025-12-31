import TinyGrad4.Basic
import TinyGrad4.Shape

namespace TinyGrad4.Kernel

/-!
# Kernel Specification Types

Generic abstractions for kernel semantics.
-/

/-- Scalar operations on a numeric type.
    Used to parameterize kernel specifications (e.g. gemm, softmax). -/
structure ScalarOps (α : Type) where
  neg : α → α
  sqrt : α → α
  reciprocal : α → α
  exp2 : α → α
  log2 : α → α
  sin : α → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  max : α → α → α
  cmplt : α → α → Bool
  where_ : Bool → α → α → α
  zero : α
  negInf : α

/-! ## Typed kernel expression language -/

inductive Ty where
  | f32
  | bool
  deriving DecidableEq, Repr

def Ty.denote : Ty → Type
  | .f32 => Float32
  | .bool => Bool

abbrev Index (shape : Shape) : Type := List Nat

inductive ReduceOp where
  | sum
  | max
  deriving DecidableEq, Repr

inductive Expr : Ty → Type where
  | input (t : Ty) (idx : Nat) : Expr t
  | constBool (b : Bool) : Expr .bool
  | constF32 (v : Float32) : Expr .f32
  | neg : Expr .f32 → Expr .f32
  | sqrt : Expr .f32 → Expr .f32
  | reciprocal : Expr .f32 → Expr .f32
  | exp2 : Expr .f32 → Expr .f32
  | log2 : Expr .f32 → Expr .f32
  | sin : Expr .f32 → Expr .f32
  | add : Expr .f32 → Expr .f32 → Expr .f32
  | sub : Expr .f32 → Expr .f32 → Expr .f32
  | mul : Expr .f32 → Expr .f32 → Expr .f32
  | div : Expr .f32 → Expr .f32 → Expr .f32
  | max : Expr .f32 → Expr .f32 → Expr .f32
  | cmplt : Expr .f32 → Expr .f32 → Expr .bool
  | where_ : Expr .bool → Expr .f32 → Expr .f32 → Expr .f32
  | truthy : Expr .f32 → Expr .bool
  deriving Repr

def evalExpr (ops : ScalarOps Float32) (read : (t : Ty) → Nat → Ty.denote t) : Expr t → Ty.denote t
  | .input t idx => read t idx
  | .constBool b => b
  | .constF32 v => v
  | .neg x => ops.neg (evalExpr ops read x)
  | .sqrt x => ops.sqrt (evalExpr ops read x)
  | .reciprocal x => ops.reciprocal (evalExpr ops read x)
  | .exp2 x => ops.exp2 (evalExpr ops read x)
  | .log2 x => ops.log2 (evalExpr ops read x)
  | .sin x => ops.sin (evalExpr ops read x)
  | .add a b => ops.add (evalExpr ops read a) (evalExpr ops read b)
  | .sub a b => ops.sub (evalExpr ops read a) (evalExpr ops read b)
  | .mul a b => ops.mul (evalExpr ops read a) (evalExpr ops read b)
  | .div a b => ops.div (evalExpr ops read a) (evalExpr ops read b)
  | .max a b => ops.max (evalExpr ops read a) (evalExpr ops read b)
  | .cmplt a b => ops.cmplt (evalExpr ops read a) (evalExpr ops read b)
  | .where_ c x y => ops.where_ (evalExpr ops read c) (evalExpr ops read x) (evalExpr ops read y)
  | .truthy x => (evalExpr ops read x).toBits != 0

private def setAt (xs : List Nat) (i v : Nat) : List Nat :=
  match xs, i with
  | [], _ => []
  | _ :: rest, 0 => v :: rest
  | x :: rest, i + 1 => x :: setAt rest i v

def mapExprF32 {shape : Shape} (ops : ScalarOps Float32) (expr : Expr .f32)
    (read : (t : Ty) → Nat → Index shape → Ty.denote t) : Index shape → Float32 :=
  fun idx => evalExpr ops (fun t i => read t i idx) expr

private def reduceInit (op : ReduceOp) (ops : ScalarOps Float32) : Float32 :=
  match op with
  | .sum => ops.zero
  | .max => ops.negInf

private def reduceStep (op : ReduceOp) (ops : ScalarOps Float32) (a b : Float32) : Float32 :=
  match op with
  | .sum => ops.add a b
  | .max => ops.max a b

partial def mapReduceKeepdimF32 {shape : Shape} (ops : ScalarOps Float32) (op : ReduceOp) (expr : Expr .f32)
    (axes : List Nat) (read : (t : Ty) → Nat → Index shape → Ty.denote t) : Index shape → Float32 :=
  let mapFn := mapExprF32 ops expr read
  let shapeList := shape
  let rec go (axs : List Nat) (idx : List Nat) : Float32 :=
    match axs with
    | [] => mapFn idx
    | ax :: rest =>
      let size := listGetD shapeList ax 1
      Id.run do
        let mut acc := reduceInit op ops
        for i in [:size] do
          let idx' := setAt idx ax i
          let v := go rest idx'
          acc := reduceStep op ops acc v
        return acc
  fun outIdx => go axes outIdx

/-- Broadcast an index from a smaller shape into a larger one. -/
def broadcastIndex (small big : Shape) (idx : Index big) : Index small :=
  let len := Nat.max small.length big.length
  let small' := List.replicate (len - small.length) 1 ++ small
  let idx' := List.replicate (len - idx.length) 0 ++ idx
  let mapped := (small'.zip idx').map (fun (s, i) => if s == 1 then 0 else i)
  mapped.drop (len - small.length)

namespace Spec

-- Additional specs can go here (e.g. gemm2D, softmax semantics).

end Spec

end TinyGrad4.Kernel
