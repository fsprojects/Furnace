// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open Furnace
open Furnace.Compose
open Furnace.Model
open Furnace.Data
open Furnace.Optim


[<TestFixture>]
type TestModelVAE() =
    
    [<Test>]
    member _.TestModelVAEMLP () =
        // Fits a little VAEMLP to structured noise
        let xdim, zdim, n = 8, 4, 16
        let m = VAEMLP(xdim*xdim, zdim)
        let x = FurnaceImage.stack(Array.init n (fun _ -> FurnaceImage.eye(xdim)*FurnaceImage.rand([xdim;xdim])))

        let lr, steps = 1e-2, 50
        let optimizer = Adam(m, lr=FurnaceImage.tensor(lr))
        let loss0 = float <| m.loss(x)
        let mutable loss = loss0
        for _ in 0..steps do
            m.reverseDiff()
            let l = m.loss(x)
            l.reverse()
            optimizer.step()
            loss <- float l

        Assert.Less(loss, loss0/2.)
    
    [<Test>]
    member _.TestModelVAE () =
        // Fits a little VAE to structured noise
        let xdim, zdim, n = 28, 4, 16
        let encoder = FurnaceImage.flatten(1) --> Linear(xdim*xdim, 8) --> FurnaceImage.relu
        let decoder = Linear(8, xdim*xdim) --> FurnaceImage.sigmoid

        let m = VAE([xdim;xdim], zdim, encoder, decoder)
        let x = FurnaceImage.stack(Array.init n (fun _ -> FurnaceImage.eye(xdim)*FurnaceImage.rand([xdim;xdim])))

        let lr, steps = 1e-2, 25
        let optimizer = Adam(m, lr=FurnaceImage.tensor(lr))
        let loss0 = float <| m.loss(x)
        let mutable loss = loss0
        for _ in 0..steps do
            m.reverseDiff()
            let l = m.loss(x)
            l.reverse()
            optimizer.step()
            loss <- float l

        Assert.Less(loss, loss0/2.)

    [<Test>]
    member _.TestModelVAEMLPSaveLoadState () =
        let xdim, zdim, n = 8, 4, 16
        let net = VAEMLP(xdim*xdim, zdim)

        let fileName = System.IO.Path.GetTempFileName()
        FurnaceImage.save(net.state, fileName) // Save pre-use
        let _ = FurnaceImage.randn([n; xdim*xdim]) --> net // Use
        net.state <- FurnaceImage.load(fileName) // Load after-use

        Assert.True(true)

    [<Test>]
    member _.TestModelVAESaveLoadState () =
        let xdim, zdim, n = 28, 4, 16
        let encoder = FurnaceImage.flatten(1) --> Linear(xdim*xdim, 8) --> FurnaceImage.relu
        let decoder = Linear(8, xdim*xdim) --> FurnaceImage.sigmoid
        let net = VAE([xdim;xdim], zdim, encoder, decoder)

        let fileName = System.IO.Path.GetTempFileName()
        FurnaceImage.save(net.state, fileName) // Save pre-use
        let _ = FurnaceImage.randn([n; xdim; xdim]) --> net // Use
        net.state <- FurnaceImage.load(fileName) // Load after-use

        Assert.True(true)