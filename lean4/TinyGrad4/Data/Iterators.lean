import Std.Data.Iterators
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.TPULoader

namespace TinyGrad4.Data

open Std.Iterators
open GPULoader
open TPULoader

/-- Device-tagged batch/element. -/
structure DeviceBatch (T : Type) where
  device : DeviceId
  value : T
  deriving Repr

/-- Device-tagged pull stream. -/
structure DeviceStream (T : Type) where
  nextFn : IO (Option (DeviceBatch T))

namespace DeviceStream

def next (s : DeviceStream T) : IO (Option (DeviceBatch T)) :=
  s.nextFn

def ofIterator (it : DataIterator T) (device : DeviceId := .cpu) : DeviceStream T :=
  {
    nextFn := do
      match ← DataIterator.next it with
      | some x => pure (some { device, value := x })
      | none => pure none
  }

def ofPrefetcher (p : Prefetcher T) (device : DeviceId := .cpu) : DeviceStream T :=
  {
    nextFn := do
      match ← Prefetcher.next p with
      | some x => pure (some { device, value := x })
      | none => pure none
  }

def ofGPU (loader : GPUDataLoader) : DeviceStream GPUBuffer :=
  {
    nextFn := do
      match ← GPUDataLoader.next loader with
      | some buf => pure (some { device := loader.device, value := buf })
      | none => pure none
  }

def ofMultiAny (pool : MultiGPULoader) : DeviceStream GPUBuffer :=
  {
    nextFn := do
      match ← MultiGPULoader.nextAny pool with
      | some (device, buf) => pure (some { device, value := buf })
      | none => pure none
  }

def ofTPU (loader : TPUDataLoader) : DeviceStream TPUBuffer :=
  {
    nextFn := do
      match ← TPUDataLoader.next loader with
      | some buf => pure (some { device := loader.device, value := buf })
      | none => pure none
  }

end DeviceStream

instance : Iterator (DeviceStream T) IO (DeviceBatch T) where
  IsPlausibleStep _ _ := True
  step it := do
    match ← it.internalState.nextFn with
    | some x =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it x (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (DeviceStream T) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (DeviceStream T) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (DeviceStream T) IO (DeviceStream T) (DeviceBatch T) :=
  ToIterator.ofM (DeviceStream T) (fun s => ⟨s⟩)

instance : ForIn IO (DeviceStream T) (DeviceBatch T) where
  forIn s init f := do
    let mut acc := init
    repeat do
      match ← s.nextFn with
      | none => break
      | some x =>
          match ← f x acc with
          | .done a => return a
          | .yield a => acc := a
    pure acc

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

namespace IteratorPrefetcher

/-- Monadic iterator view for `IteratorPrefetcher`. -/
def iterM (p : IteratorPrefetcher T) : IterM (α := IteratorPrefetcher T) IO T :=
  ⟨p⟩

end IteratorPrefetcher

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

instance : Iterator (IteratorPrefetcher T) IO T where
  IsPlausibleStep _ _ := True
  step it := do
    match ← IteratorPrefetcher.next it.internalState with
    | some x =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it x (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (IteratorPrefetcher T) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (IteratorPrefetcher T) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (IteratorPrefetcher T) IO (IteratorPrefetcher T) T :=
  ToIterator.ofM (IteratorPrefetcher T) (fun p => ⟨p⟩)

instance : ForIn IO (IteratorPrefetcher T) T where
  forIn p init f := do
    let mut acc := init
    repeat do
      match ← IteratorPrefetcher.next p with
      | none => break
      | some x =>
          match ← f x acc with
          | .done a =>
              IteratorPrefetcher.cancel p
              return a
          | .yield a => acc := a
    pure acc

namespace GPUDataLoader

/-- Monadic iterator view for `GPUDataLoader`. -/
def iterM (loader : GPUDataLoader) : IterM (α := GPUDataLoader) IO GPUBuffer :=
  ⟨loader⟩

end GPUDataLoader

namespace TPUDataLoader

/-- Monadic iterator view for `TPUDataLoader`. -/
def iterM (loader : TPUDataLoader) : IterM (α := TPUDataLoader) IO TPUBuffer :=
  ⟨loader⟩

end TPUDataLoader

instance : Iterator TPUDataLoader IO TPUBuffer where
  IsPlausibleStep _ _ := True
  step it := do
    match ← TPUDataLoader.next it.internalState with
    | some buf =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it buf (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop TPUDataLoader IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect TPUDataLoader IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator TPUDataLoader IO TPUDataLoader TPUBuffer :=
  ToIterator.ofM TPUDataLoader (fun loader => ⟨loader⟩)

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

namespace MultiGPULoader

/-- Monadic iterator view for `MultiGPULoader` (sync batches). -/
def iterM (pool : MultiGPULoader) :
    IterM (α := MultiGPULoader) IO (Array (DeviceBatch GPUBuffer)) :=
  ⟨pool⟩

end MultiGPULoader

instance : Iterator MultiGPULoader IO (Array (DeviceBatch GPUBuffer)) where
  IsPlausibleStep _ _ := True
  step it := do
    let batches ← MultiGPULoader.nextAll it.internalState
    if batches.all (fun (_, b) => b.isNone) then
      pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)
    else
      let actual := batches.filterMap fun (device, buf) =>
        buf.map fun b => { device, value := b }
      pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it actual (by trivial)

instance : IteratorLoop MultiGPULoader IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect MultiGPULoader IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator MultiGPULoader IO MultiGPULoader (Array (DeviceBatch GPUBuffer)) :=
  ToIterator.ofM MultiGPULoader (fun pool => ⟨pool⟩)

instance : ForIn IO MultiGPULoader (Array (DeviceBatch GPUBuffer)) where
  forIn pool init f := do
    let mut acc := init
    repeat do
      let batches ← MultiGPULoader.nextAll pool
      if batches.all (fun (_, b) => b.isNone) then
        break
      let actual := batches.filterMap fun (device, buf) =>
        buf.map fun b => { device, value := b }
      match ← f actual acc with
      | .done a => return a
      | .yield a => acc := a
    pure acc

end TinyGrad4.Data
