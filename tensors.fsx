#r "nuget: Furnace-lite,1.0.9"
(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fsprojects/Furnace/blob/master/tensors.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/fsprojects/Furnace/master?filepath=tensors.ipynb)&emsp;
[![Script](img/badge-script.svg)](tensors.fsx)&emsp;
[![Script](img/badge-notebook.svg)](tensors.ipynb)

* The [FurnaceImage](https://fsprojects.github.io/Furnace/reference/furnace-furnaceimage.html) API
  

* The [Tensor](https://fsprojects.github.io/Furnace/reference/furnace-tensor.html) type
  

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

