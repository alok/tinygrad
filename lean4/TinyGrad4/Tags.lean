namespace TinyGrad4

structure Tagger (ι : Type) (α : Type) where
  apply : ι → α → α

instance : GetElem (Tagger ι α) ι (α → α) (fun _ _ => True) where
  getElem t i _ := t.apply i

def fusion {α} : Tagger String α :=
  { apply := fun _ x => x }

def cost {α} : Tagger Nat α :=
  { apply := fun _ x => x }

def device {α} : Tagger String α :=
  { apply := fun _ x => x }

end TinyGrad4
