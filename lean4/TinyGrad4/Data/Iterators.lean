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

def ofGPU (loader : GPUDataLoader batch itemShape dtype) :
    DeviceStream (GPULease (batch :: itemShape) dtype) :=
  {
    nextFn := do
      match ← GPUDataLoader.next loader with
      | some buf => pure (some { device := loader.device, value := buf })
      | none => pure none
  }

def ofMultiAny (pool : MultiGPULoader batch itemShape dtype) :
    DeviceStream (GPULease (batch :: itemShape) dtype) :=
  {
    nextFn := do
      match ← MultiGPULoader.nextAny pool with
      | some (device, buf) => pure (some { device, value := buf })
      | none => pure none
  }

def ofTPU (loader : TPUDataLoader batch itemShape dtype) :
    DeviceStream (TPUBuffer (batch :: itemShape) dtype) :=
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

namespace BatchPrefetcher

/-- Monadic iterator view for `BatchPrefetcher`. -/
def iterM (p : BatchPrefetcher B) : IterM (α := BatchPrefetcher B) IO B :=
  ⟨p⟩

end BatchPrefetcher

namespace MultiIteratorPrefetcher

/-- Monadic iterator view for `MultiIteratorPrefetcher`. -/
def iterM (p : MultiIteratorPrefetcher T) : IterM (α := MultiIteratorPrefetcher T) IO T :=
  ⟨p⟩

end MultiIteratorPrefetcher

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

instance : Iterator (BatchPrefetcher B) IO B where
  IsPlausibleStep _ _ := True
  step it := do
    match ← BatchPrefetcher.next it.internalState with
    | some x =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it x (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (BatchPrefetcher B) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (BatchPrefetcher B) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (BatchPrefetcher B) IO (BatchPrefetcher B) B :=
  ToIterator.ofM (BatchPrefetcher B) (fun p => ⟨p⟩)

instance : Iterator (MultiIteratorPrefetcher T) IO T where
  IsPlausibleStep _ _ := True
  step it := do
    match ← MultiIteratorPrefetcher.next it.internalState with
    | some x =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it x (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (MultiIteratorPrefetcher T) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (MultiIteratorPrefetcher T) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (MultiIteratorPrefetcher T) IO (MultiIteratorPrefetcher T) T :=
  ToIterator.ofM (MultiIteratorPrefetcher T) (fun p => ⟨p⟩)

namespace GPUDataLoader

/-- Monadic iterator view for `GPUDataLoader`. -/
def iterM (loader : GPUDataLoader batch itemShape dtype) :
    IterM (α := GPUDataLoader batch itemShape dtype) IO (GPULease (batch :: itemShape) dtype) :=
  ⟨loader⟩

end GPUDataLoader

namespace TPUDataLoader

/-- Monadic iterator view for `TPUDataLoader`. -/
def iterM (loader : TPUDataLoader batch itemShape dtype) :
    IterM (α := TPUDataLoader batch itemShape dtype) IO (TPUBuffer (batch :: itemShape) dtype) :=
  ⟨loader⟩

end TPUDataLoader

instance : Iterator (TPUDataLoader batch itemShape dtype) IO (TPUBuffer (batch :: itemShape) dtype) where
  IsPlausibleStep _ _ := True
  step it := do
    match ← TPUDataLoader.next it.internalState with
    | some buf =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it buf (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (TPUDataLoader batch itemShape dtype) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (TPUDataLoader batch itemShape dtype) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (TPUDataLoader batch itemShape dtype) IO
    (TPUDataLoader batch itemShape dtype) (TPUBuffer (batch :: itemShape) dtype) :=
  ToIterator.ofM (TPUDataLoader batch itemShape dtype) (fun loader => ⟨loader⟩)

instance : Iterator (GPUDataLoader batch itemShape dtype) IO (GPULease (batch :: itemShape) dtype) where
  IsPlausibleStep _ _ := True
  step it := do
    match ← GPUDataLoader.next it.internalState with
    | some buf =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it buf (by trivial)
    | none =>
        pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)

instance : IteratorLoop (GPUDataLoader batch itemShape dtype) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (GPUDataLoader batch itemShape dtype) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (GPUDataLoader batch itemShape dtype) IO
    (GPUDataLoader batch itemShape dtype) (GPULease (batch :: itemShape) dtype) :=
  ToIterator.ofM (GPUDataLoader batch itemShape dtype) (fun loader => ⟨loader⟩)

namespace MultiGPULoader

/-- Monadic iterator view for `MultiGPULoader` (sync batches). -/
def iterM (pool : MultiGPULoader batch itemShape dtype) :
    IterM (α := MultiGPULoader batch itemShape dtype) IO
      (Array (DeviceBatch (GPULease (batch :: itemShape) dtype))) :=
  ⟨pool⟩

end MultiGPULoader

instance : Iterator (MultiGPULoader batch itemShape dtype) IO
    (Array (DeviceBatch (GPULease (batch :: itemShape) dtype))) where
  IsPlausibleStep _ _ := True
  step it := do
    let batches ← MultiGPULoader.nextAll it.internalState
    if batches.all (fun (_, b) => b.isNone) then
      pure <| Std.Shrink.deflate <| PlausibleIterStep.done (by trivial)
    else
      let actual := batches.filterMap fun (device, buf) =>
        buf.map fun b => { device, value := b }
      pure <| Std.Shrink.deflate <| PlausibleIterStep.yield it actual (by trivial)

instance : IteratorLoop (MultiGPULoader batch itemShape dtype) IO IO :=
  IteratorLoop.defaultImplementation

instance : IteratorCollect (MultiGPULoader batch itemShape dtype) IO IO :=
  IteratorCollect.defaultImplementation

instance : ToIterator (MultiGPULoader batch itemShape dtype) IO
    (MultiGPULoader batch itemShape dtype)
    (Array (DeviceBatch (GPULease (batch :: itemShape) dtype))) :=
  ToIterator.ofM (MultiGPULoader batch itemShape dtype) (fun pool => ⟨pool⟩)

instance : ForIn IO (MultiGPULoader batch itemShape dtype)
    (Array (DeviceBatch (GPULease (batch :: itemShape) dtype))) where
  forIn pool init f := do
    let mut acc := init
    repeat do
      let batches ← MultiGPULoader.nextAll pool
      if batches.all (fun (_, b) => b.isNone) then
        break
      let actual := batches.filterMap fun (device, buf) =>
        buf.map fun b => { device, value := b }
      match ← f actual acc with
      | .done a =>
          for batch in actual do
            batch.value.release
          return a
      | .yield a =>
          for batch in actual do
            batch.value.release
          acc := a
    pure acc

end TinyGrad4.Data
