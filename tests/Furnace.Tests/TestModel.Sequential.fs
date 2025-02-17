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
type TestModelSequential () =

    [<Test>]
    member _.TestModelSequential () =
        let m1 = Linear(1, 2)
        let m2 = Linear(2, 3)
        let m3 = Linear(3, 4)

        let m = m1 --> m2 --> m3
        let mSequential = Sequential([m1;m2;m3])

        let x = FurnaceImage.randn([1;1])
        let y = x --> m
        let ySequential = x --> mSequential

        Assert.True(ySequential.allclose(y))

    [<Test>]
    member _.TestModelSequentialSaveLoadState () =
        let batchSize = 4
        let inFeatures = 1
        let m1 = Linear(1, 2)
        let m2 = Linear(2, 3)
        let m3 = Linear(3, 4)
        let net = Sequential([m1;m2;m3])

        let fileName = System.IO.Path.GetTempFileName()
        FurnaceImage.save(net.state, fileName)
        let _ = FurnaceImage.randn([batchSize; inFeatures]) --> net
        net.state <- FurnaceImage.load(fileName)
        Assert.True(true)