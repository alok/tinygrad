import Std

/-! Run id generation helpers. -/

namespace Wandb.RunId

def alphabet : Array Char :=
  #['a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z','0','1','2','3',
    '4','5','6','7','8','9']

/-- Generate a random base-36 run id using {lit}`IO.getRandomBytes`. -/
def generate (length : Nat := 8) : IO String := do
  let bytes ← IO.getRandomBytes (USize.ofNat length)
  let mut out := ""
  for i in [0:length] do
    let b := bytes.get! i
    let idx := (UInt8.toNat b) % alphabet.size
    out := out.push (alphabet.getD idx 'a')
  pure out

/-- Faster run id using {lit}`IO.rand` (not cryptographically strong). -/
def generateFast (length : Nat := 8) : IO String := do
  let mut out := ""
  for _ in [0:length] do
    let idx ← IO.rand 0 (alphabet.size - 1)
    out := out.push (alphabet.getD idx 'a')
  pure out

end Wandb.RunId
