(**
Test

*)
open Furnace

FurnaceImage.config(backend=Backend.Reference)

let a = FurnaceImage.tensor([1,2,3])
printfn "%A" a(* output: 
input.fsx (1,6)-(1,13) typecheck error The namespace or module 'Furnace' is not defined.
input.fsx (3,1)-(3,13) typecheck error The value, namespace, type or module 'FurnaceImage' is not defined.
input.fsx (5,9)-(5,21) typecheck error The value, namespace, type or module 'FurnaceImage' is not defined.*)

