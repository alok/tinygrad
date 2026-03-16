import Float64
namespace TinyGrad4

structure Tagger (ι : Type) (α : Type) where
  apply : ι → α → α

instance : GetElem (Tagger ι α) ι (α → α) (fun _ _ => True) where
  getElem t i _ := t.apply i

class FusionTaggable (α : Type) where
  applyFusion : String → α → α

class CostTaggable (α : Type) where
  applyCost : Nat → α → α

class DeviceTaggable (α : Type) where
  applyDevice : String → α → α

class AxisTaggable (α : Type) where
  applyAxis : Nat → α → α

instance (priority := low) : FusionTaggable α where
  applyFusion _ x := x

instance (priority := low) : CostTaggable α where
  applyCost _ x := x

instance (priority := low) : DeviceTaggable α where
  applyDevice _ x := x

instance (priority := low) : AxisTaggable α where
  applyAxis _ x := x

def fusion {α} [FusionTaggable α] : Tagger String α :=
  { apply := FusionTaggable.applyFusion }

def cost {α} [CostTaggable α] : Tagger Nat α :=
  { apply := CostTaggable.applyCost }

def device {α} [DeviceTaggable α] : Tagger String α :=
  { apply := DeviceTaggable.applyDevice }

def axis {α} [AxisTaggable α] : Tagger Nat α :=
  { apply := AxisTaggable.applyAxis }

end TinyGrad4
