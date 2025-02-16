(**
Test

*)
open Furnace

FurnaceImage.config(backend=Backend.Reference)

let a = FurnaceImage.tensor([1,2,3])
printfn "%A" a(* output: 
*)

