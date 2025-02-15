(*** condition: prepare ***)
#I "../tests/Furnace.Tests/bin/Debug/net6.0"
#r "Furnace.Core.dll"
#r "Furnace.Data.dll"
#r "Furnace.Backends.Reference.dll"
#r "Furnace.Backends.Torch.dll"
// These are needed to make fsdocs --eval work. If we don't select a backend like this in the beginning, we get erratic behavior.
Furnace.FurnaceImage.config(backend=Furnace.Backend.Reference)
Furnace.FurnaceImage.seed(123)

(**
Test 
*)

open Furnace

FurnaceImage.config(backend=Backend.Reference)

let a = FurnaceImage.tensor([1,2,3])
printfn "%A" a
(*** include-fsi-output ***)