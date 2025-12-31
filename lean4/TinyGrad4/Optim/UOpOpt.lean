import TinyGrad4.UOp.UOp

namespace TinyGrad4.Optim

/-- Optimization placeholder: returns roots unchanged. -/
def optimizeKeepUids (roots : List TinyGrad4.UOp) : List TinyGrad4.UOp :=
  roots

end TinyGrad4.Optim
