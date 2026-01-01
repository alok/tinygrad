import Std.Data.Iterators
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.GPULoader

namespace TinyGrad4.Data

open Std.Iterators
open GPULoader

namespace DataIterator

/-- Monadic iterator view for `DataIterator`. -/
def iterM (it : DataIterator T) : IterM (α := DataIterator T) IO T :=
  ⟨it⟩

end DataIterator

instance : Iterator (DataIterator T) IO T where
  IsPlausibleStep _ _ := True
  step it := do
    match ← it.internalState.nextFn with
    | some x =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it x (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (DataIterator T) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (DataIterator T) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (DataIterator T) IO (DataIterator T) T :=
  ToIterator.ofM (DataIterator T) (fun it => ⟨it⟩)

instance : ForIn IO (DataIterator T) T where
  forIn it init f := do
    let mut acc := init
    repeat do
      match ← DataIterator.next it with
      | none => break
      | some x =>
          match ← f x acc with
          | .done a => return a
          | .yield a => acc := a
    pure acc

namespace Prefetcher

/-- Monadic iterator view for `Prefetcher`. -/
def iterM (p : Prefetcher T) : IterM (α := Prefetcher T) IO T :=
  ⟨p⟩

end Prefetcher

instance : Iterator (Prefetcher T) IO T where
  IsPlausibleStep _ _ := True
  step it := do
    match ← Prefetcher.next it.internalState with
    | some x =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it x (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (Prefetcher T) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (Prefetcher T) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (Prefetcher T) IO (Prefetcher T) T :=
  ToIterator.ofM (Prefetcher T) (fun p => ⟨p⟩)

instance : ForIn IO (Prefetcher T) T where
  forIn p init f := do
    let mut acc := init
    repeat do
      match ← Prefetcher.next p with
      | none => break
      | some x =>
          match ← f x acc with
          | .done a => return a
          | .yield a => acc := a
    pure acc

namespace GPUDataLoader

/-- Monadic iterator view for `GPUDataLoader`. -/
def iterM (loader : GPUDataLoader) : IterM (α := GPUDataLoader) IO GPUBuffer :=
  ⟨loader⟩

end GPUDataLoader

instance : Iterator GPUDataLoader IO GPUBuffer where
  IsPlausibleStep _ _ := True
  step it := do
    match ← GPUDataLoader.next it.internalState with
    | some buf =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it buf (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop GPUDataLoader IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect GPUDataLoader IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator GPUDataLoader IO GPUDataLoader GPUBuffer :=
  ToIterator.ofM GPUDataLoader (fun loader => ⟨loader⟩)

end TinyGrad4.Data
