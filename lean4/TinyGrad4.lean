import TinyGrad4.Basic
import TinyGrad4.Linter.FloatLinter
import TinyGrad4.DType
import TinyGrad4.Shape
import TinyGrad4.Data.ArrayN
import TinyGrad4.Data.Loader
import TinyGrad4.Ops
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Notation
import TinyGrad4.Tensor.Movement
import TinyGrad4.Tensor.Math
import TinyGrad4.Tags
import TinyGrad4.Debug
import TinyGrad4.Pretty
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Backend.Schedule
import TinyGrad4.Backend.Rangeify
import TinyGrad4.Backend.Memory
import TinyGrad4.Backend.FusedEwiseExpr
import TinyGrad4.Backend.FusedReduceExpr
import TinyGrad4.Backend.FusedSoftmaxExpr
import TinyGrad4.Backend.FusedMatmulExpr
import TinyGrad4.Kernel.Spec
import TinyGrad4.Kernel.Trusted
import TinyGrad4.Gradient.Rules
import TinyGrad4.Gradient.Autodiff
import TinyGrad4.Optim.SGD
import TinyGrad4.Optim.Init
import TinyGrad4.Optim.Optimizer
import TinyGrad4.Optim.UOpOpt

-- Tests are not imported here to avoid circular dependencies
-- Run tests with: lake build TinyGrad4.Test.GradientCheck TinyGrad4.Test.MLP
