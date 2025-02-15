(*** condition: prepare ***)
#I "../tests/Furnace.Tests/bin/Debug/net6.0"
#r "Furnace.Core.dll"
#r "Furnace.Data.dll"
#r "Furnace.Backends.Reference.dll"
#r "Furnace.Backends.Torch.dll"
// These are needed to make fsdocs --eval work. If we don't select a backend like this in the beginning, we get erratic behavior.
Furnace.FurnaceImage.config(backend=Furnace.Backend.Reference)
Furnace.FurnaceImage.seed(123)

(*** condition: fsx ***)
#if FSX
#r "nuget: Furnace-lite,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
// Google Colab only: uncomment and run the following to install dotnet and the F# kernel
// !bash <(curl -Ls https://raw.githubusercontent.com/gbaydin/scripts/main/colab_dotnet6.sh)
#endif // IPYNB
(*** condition: ipynb ***)
#if IPYNB
// Import Furnace package
#r "nuget: Furnace-lite,{{fsdocs-package-version}}"

// Set dotnet interactive formatter to plaintext
Formatter.SetPreferredMimeTypesFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fsprojects/furnace/blob/master/{{fsdocs-source-basename}}.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/fsprojects/furnace/master?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

* The `cref:T:Furnace.FurnaceImage` API

* The `cref:T:Furnace.Tensor` type

Saving tensors as image and loading images as tensors


## Converting between Tensors and arrays

System.Array and F# arrays

*)

open Furnace

// Tensor
let t1 = FurnaceImage.tensor [ 0.0 .. 0.2 .. 1.0 ]

// System.Array
let a1 = t1.toArray()

// []<float32>
let a1b = t1.toArray() :?> float32[]

// Tensor
let t2 = FurnaceImage.randn([3;3;3])

// [,,]<float32>
let a2 = t2.toArray() :?> float32[,,]