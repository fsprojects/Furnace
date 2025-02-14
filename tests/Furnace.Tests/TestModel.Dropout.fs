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
type TestModelDropout () =

    [<Test>]
    member _.TestModelDropout () =
        let m = Dropout(1.)
        let x = FurnaceImage.randn([10;10])
        Assert.CheckEqual(m.parametersVector.shape, [| 0 |])
        m.train()
        let xtrain = x --> m
        Assert.CheckEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.CheckEqual(x, xeval)

    [<Test>]
    member _.TestModelDropout2d () =
        let m = Dropout2d(1.)
        let x = FurnaceImage.randn([10;4;10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.CheckEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.CheckEqual(x, xeval)

    [<Test>]
    member _.TestModelDropout3d () =
        let m = Dropout3d(1.)
        let x = FurnaceImage.randn([10;4;10;10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.CheckEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.CheckEqual(x, xeval)

    [<Test>]
    member _.TestModelDropoutaveLoadState () =
        let net = Dropout(0.5)

        let fileName = System.IO.Path.GetTempFileName()
        FurnaceImage.save(net.state, fileName) // Save pre-use
        let _ = FurnaceImage.randn([10; 10]) --> net // Use
        net.state <- FurnaceImage.load(fileName) // Load after-use

        Assert.True(true)

    [<Test>]
    member _.TestModelDropout2daveLoadState () =
        let net = Dropout2d(0.5)

        let fileName = System.IO.Path.GetTempFileName()
        FurnaceImage.save(net.state, fileName) // Save pre-use
        let _ = FurnaceImage.randn([10; 10; 10; 10]) --> net // Use
        net.state <- FurnaceImage.load(fileName) // Load after-use

        Assert.True(true)

    [<Test>]
    member _.TestModelDropout3daveLoadState () =
        let net = Dropout3d(0.5)

        let fileName = System.IO.Path.GetTempFileName()
        FurnaceImage.save(net.state, fileName) // Save pre-use
        let _ = FurnaceImage.randn([10; 10; 10; 10; 10]) --> net // Use
        net.state <- FurnaceImage.load(fileName) // Load after-use

        Assert.True(true)
