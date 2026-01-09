import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math

namespace TinyGrad4

instance {s1 s2 : Shape} {d : DType} :
    HAdd (StaticTensor s1 d) (StaticTensor s2 d) (TensorM (StaticTensor (Shape.broadcastOut s1 s2) d)) where
  hAdd a b := StaticTensor.addB a b

instance {s1 s2 : Shape} {d : DType} :
    HSub (StaticTensor s1 d) (StaticTensor s2 d) (TensorM (StaticTensor (Shape.broadcastOut s1 s2) d)) where
  hSub a b := StaticTensor.subB a b

instance {s1 s2 : Shape} {d : DType} :
    HMul (StaticTensor s1 d) (StaticTensor s2 d) (TensorM (StaticTensor (Shape.broadcastOut s1 s2) d)) where
  hMul a b := StaticTensor.mulB a b

instance {s1 s2 : Shape} {d : DType} :
    HDiv (StaticTensor s1 d) (StaticTensor s2 d) (TensorM (StaticTensor (Shape.broadcastOut s1 s2) d)) where
  hDiv a b := StaticTensor.divB a b

instance {s1 s2 : Shape} {d : DType} :
    HPow (StaticTensor s1 d) (StaticTensor s2 d) (TensorM (StaticTensor (Shape.broadcastOut s1 s2) d)) where
  hPow a b := StaticTensor.powB a b

end TinyGrad4
