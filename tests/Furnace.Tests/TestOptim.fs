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
type TestOptim () =
    do FurnaceImage.seed(123)
    let n, din, dout = 64, 100, 10
    let inputs  = FurnaceImage.randn([n; din])
    let targets = FurnaceImage.randn([n; dout])
    let dataset = TensorDataset(inputs, targets)
    let dataloader = dataset.loader(8, shuffle=true)

    let rosenbrock (x:Tensor) = 
        let x, y = x[0], x[1]
        (1. - x)**2 + 100. * (y - x**2)**2

    [<Test>]
    member _.TestOptimizerStep () =
        let net = Linear(din, dout)
        let optimizer = SGD(net)
        let step0 = optimizer.stateStep
        let step0Correct = 0
        net.reverseDiff()
        let y = net.forward(inputs)
        let loss = FurnaceImage.mseLoss(y, targets)
        loss.reverse()
        optimizer.step()
        let step1 = optimizer.stateStep
        let step1Correct = 1
        Assert.AreEqual(step0Correct, step0)
        Assert.AreEqual(step1Correct, step1)

    [<Test>]
    member _.TestOptimModelSGDStyle1 () =
        // Trains a linear regressor
        let net = Linear(din, dout)
        let lr, mom, epochs = 1e-2, 0.9, 250
        let optimizer = SGD(net, lr=FurnaceImage.tensor(lr), momentum=FurnaceImage.tensor(mom), nesterov=true)
        for _ in 0..epochs do
            for _, inputs, targets in dataloader.epoch() do
                net.reverseDiff()
                let y = net.forward(inputs)
                let loss = FurnaceImage.mseLoss(y, targets)
                loss.reverse()
                optimizer.step()
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestOptimModelSGDStyle2 () =
        // Trains a linear regressor
        let net = Linear(din, dout)
        let lr, mom, epochs = 1e-2, 0.9, 250
        optim.sgd(net, dataloader, FurnaceImage.mseLoss, lr=FurnaceImage.tensor(lr), momentum=FurnaceImage.tensor(mom), nesterov=true,  threshold=1e-4, epochs=epochs)
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestOptimModelSGDStyle3 () =
        // Trains a linear regressor
        let net = Linear(din, dout)
        let lr, epochs = 1e-1, 250
        for _ in 0..epochs do
            for _, inputs, targets in dataloader.epoch() do
                let loss p = net.asFunction p inputs |> FurnaceImage.mseLoss targets
                let g = FurnaceImage.grad loss net.parametersVector
                net.parametersVector <- net.parametersVector - lr * g

        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestOptimModelAdamStyle1 () =
        // Trains a linear regressor
        let net = Linear(din, dout)
        let lr, epochs = 1e-2, 50
        let optimizer = Adam(net, lr=FurnaceImage.tensor(lr))
        for _ in 0..epochs do
            for _, inputs, targets in dataloader.epoch() do
                net.reverseDiff()
                let y = net.forward(inputs)
                let loss = FurnaceImage.mseLoss(y, targets)
                loss.reverse()
                optimizer.step()
                // printfn "%A" (float loss)
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestOptimModelAdamStyle2 () =
        // Trains a linear regressor
        let net = Linear(din, dout)
        let lr, epochs = 1e-2, 50
        optim.adam(net, dataloader, FurnaceImage.mseLoss, lr=FurnaceImage.tensor(lr), threshold=1e-4, epochs=epochs)
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestOptimFunSGD () =
        let x0 = FurnaceImage.tensor([1.5, 1.5])
        let lr, momentum, iters, threshold = 1e-3, 0.5, 1000, 1e-3
        let fx, x = optim.sgd(rosenbrock, x0, lr=FurnaceImage.tensor(lr), momentum=FurnaceImage.tensor(momentum), nesterov=true, iters=iters, threshold=threshold)
        let fxOpt = FurnaceImage.tensor(0.)
        let xOpt = FurnaceImage.tensor([1., 1.])
        Assert.True(fxOpt.allclose(fx, 0.1, 0.1))
        Assert.True(xOpt.allclose(x, 0.1, 0.1))

    [<Test>]
    member _.TestOptimFunAdam () =
        let x0 = FurnaceImage.tensor([1.5, 1.5])
        let lr, iters, threshold = 1., 1000, 1e-3
        let fx, x = optim.adam(rosenbrock, x0, lr=FurnaceImage.tensor(lr), iters=iters, threshold=threshold)
        let fxOpt = FurnaceImage.tensor(0.)
        let xOpt = FurnaceImage.tensor([1., 1.])
        Assert.True(fxOpt.allclose(fx, 0.1, 0.1))
        Assert.True(xOpt.allclose(x, 0.1, 0.1))        