// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open NUnit.Framework
open Furnace

[<TestFixture>]
type TestBranchCoverage() =

    [<Test>]
    member _.TestTensorCastingBranches() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t = combo.tensor([1.0f; 2.0f; 3.0f])
        
        // Test cast to same type (should return same tensor)
        let sameCast = t.cast(Dtype.Float32)
        Assert.AreSame(t, sameCast)
        
        // Test different type casts
        let intCast = t.cast(Dtype.Int32)
        Assert.AreEqual(Dtype.Int32, intCast.dtype)
        
        let doubleCast = t.cast(Dtype.Float64)
        Assert.AreEqual(Dtype.Float64, doubleCast.dtype)

    [<Test>]
    member _.TestTensorBackendMoveBranches() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t = combo.tensor([1.0f; 2.0f; 3.0f])
        
        // Test move to same backend (should return same tensor)
        let sameBackend = t.move(Backend.Reference)
        Assert.AreSame(t, sameBackend)

    [<Test>]
    member _.TestTensorDeviceMoveBranches() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t = combo.tensor([1.0f; 2.0f; 3.0f])
        
        // Test move to same device (should return same tensor)  
        let sameDevice = t.move(Device.CPU)
        Assert.AreSame(t, sameDevice)

    [<Test>]
    member _.TestGenericCastingBranches() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t = combo.tensor([1.0f; 2.0f; 3.0f])
        
        // Test different generic cast types
        let float32Cast = t.cast<float32>()
        Assert.AreEqual(Dtype.Float32, float32Cast.dtype)
        
        let float64Cast = t.cast<double>()
        Assert.AreEqual(Dtype.Float64, float64Cast.dtype)
        
        let int32Cast = t.cast<int32>()
        Assert.AreEqual(Dtype.Int32, int32Cast.dtype)
        
        let int64Cast = t.cast<int64>()
        Assert.AreEqual(Dtype.Int64, int64Cast.dtype)
        
        let int16Cast = t.cast<int16>()
        Assert.AreEqual(Dtype.Int16, int16Cast.dtype)
        
        let int8Cast = t.cast<int8>()
        Assert.AreEqual(Dtype.Int8, int8Cast.dtype)
        
        let bytecast = t.cast<byte>()
        Assert.AreEqual(Dtype.Byte, bytecast.dtype)
        
        let boolCast = t.cast<bool>()
        Assert.AreEqual(Dtype.Bool, boolCast.dtype)

    [<Test>]
    member _.TestInvalidGenericCastBranch() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t = combo.tensor([1.0f; 2.0f; 3.0f])
        
        // Test invalid cast type should throw
        Assert.Throws<System.Exception>(fun () -> 
            t.cast<string>() |> ignore) |> ignore

    [<Test>]
    member _.TestTensorComparisons() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t1 = combo.tensor([1.0f; 2.0f; 3.0f])
        let t2 = combo.tensor([1.0f; 2.0f; 3.0f])
        let t3 = combo.tensor([1.0f; 2.0f; 4.0f])
        
        // Test equality branches
        Assert.True(t1.Equals(t2))
        Assert.False(t1.Equals(t3))
        Assert.False(t1.Equals(null))

    [<Test>]
    member _.TestTensorShapeValidation() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        
        // Test zero-sized tensors
        let empty = FurnaceImage.zeros([0], dtype=combo.dtype, backend=combo.backend, device=combo.device)
        Assert.AreEqual(0, empty.nelement)
        
        // Test single element tensors
        let single = combo.tensor([42.0f])
        Assert.AreEqual(1, single.nelement)

    [<Test>]
    member _.TestTensorIndexingBoundaries() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t = combo.tensor([1.0f; 2.0f; 3.0f; 4.0f; 5.0f])
        
        // Test valid indexing
        Assert.DoesNotThrow(fun () -> t[0] |> ignore)
        Assert.DoesNotThrow(fun () -> t[4] |> ignore)
        
        // Test boundary conditions for slicing
        let slice1 = t[0..2]
        Assert.AreEqual([|3|], slice1.shape)
        
        let slice2 = t[1..4]
        Assert.AreEqual([|4|], slice2.shape)

    [<Test>]
    member _.TestTensorOperationEdgeCases() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        
        // Test operations with different sized tensors 
        let t1x1 = combo.tensor([[1.0f]])
        let t2x2 = combo.tensor([[1.0f; 2.0f]; [3.0f; 4.0f]])
        
        // Test broadcasting operations
        let broadcast1 = t1x1 + t2x2
        Assert.AreEqual([|2; 2|], broadcast1.shape)
        
        let broadcast2 = t2x2 * combo.tensor([2.0f])
        Assert.AreEqual([|2; 2|], broadcast2.shape)

    [<Test>]
    member _.TestBoolTensorOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test boolean tensor creation and operations
        let b1 = combo.tensor([true; false; true])
        let b2 = combo.tensor([false; true; true])
        
        // Test boolean operations that may have specific branches
        Assert.AreEqual(Dtype.Bool, b1.dtype)
        Assert.AreEqual(3, b1.nelement)
        
        // Test boolean comparisons
        let eq = b1.eq(b2)
        Assert.AreEqual(Dtype.Bool, eq.dtype)

    [<Test>]
    member _.TestTensorCreationEdgeCases() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        
        // Test creating tensors from different input types
        let fromInt = combo.tensor([1; 2; 3])
        Assert.AreEqual(Dtype.Float32, fromInt.dtype) // Should be cast to combo dtype
        
        let fromFloat = combo.tensor([1.0; 2.0; 3.0])
        Assert.AreEqual(Dtype.Float32, fromFloat.dtype)
        
        // Test nested arrays
        let nested = combo.tensor([[[1.0f]]])
        Assert.AreEqual([|1; 1; 1|], nested.shape)
        Assert.AreEqual(3, nested.dim)

    [<Test>]
    member _.TestTensorMemoryLayout() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        
        // Test different memory layouts and shapes
        let t1d = combo.tensor([1.0f; 2.0f; 3.0f; 4.0f])
        let t2d = t1d.view([2; 2])
        
        Assert.AreEqual([|4|], t1d.shape)
        Assert.AreEqual([|2; 2|], t2d.shape)
        Assert.AreEqual(t1d.nelement, t2d.nelement)
        
        // Test transpose operations
        let transposed = t2d.transpose()
        Assert.AreEqual([|2; 2|], transposed.shape)

    [<Test>]
    member _.TestTensorReductionEdgeCases() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        
        // Test reductions on different tensor shapes
        let t1d = combo.tensor([1.0f; 2.0f; 3.0f; 4.0f])
        let sum1d = t1d.sum()
        Assert.True(abs(sum1d.toScalar().toSingle() - 10.0f) < 0.001f)
        
        let t2d = combo.tensor([[1.0f; 2.0f]; [3.0f; 4.0f]])
        let sum2d = t2d.sum()
        Assert.True(abs(sum2d.toScalar().toSingle() - 10.0f) < 0.001f)
        
        // Test specific dimension reductions
        let sumDim0 = t2d.sum(0)
        Assert.AreEqual([|2|], sumDim0.shape)
        
        let sumDim1 = t2d.sum(1)
        Assert.AreEqual([|2|], sumDim1.shape)

    [<Test>]
    member _.TestLowPrecisionTypeOperations() =
        // Test BFloat16 specific operations for branch coverage
        let comboBF16 = ComboInfo(Backend.Reference, Device.CPU, Dtype.BFloat16)
        let t1 = comboBF16.tensor([1.0f; 2.0f; 3.0f])
        let t2 = comboBF16.tensor([2.0f; 3.0f; 4.0f])
        
        // Test various comparison operations
        let eq = t1.eq(t2)
        let ne = t1.ne(t2)
        let lt = t1.lt(t2)
        let le = t1.le(t2)
        let gt = t1.gt(t2)
        let ge = t1.ge(t2)
        
        Assert.AreEqual(Dtype.Bool, eq.dtype)
        Assert.AreEqual(Dtype.Bool, ne.dtype)
        Assert.AreEqual(Dtype.Bool, lt.dtype)
        Assert.AreEqual(Dtype.Bool, le.dtype)
        Assert.AreEqual(Dtype.Bool, gt.dtype)
        Assert.AreEqual(Dtype.Bool, ge.dtype)

    [<Test>]
    member _.TestActivationFunctionBranches() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        let t = combo.tensor([-2.0f; -1.0f; 0.0f; 1.0f; 2.0f])
        
        // Test various activation functions that might have different code paths
        let sigmoid = t.sigmoid()
        Assert.AreEqual(5, sigmoid.nelement)
        Assert.AreEqual(Dtype.Float32, sigmoid.dtype)
        
        let tanh = t.tanh()
        Assert.AreEqual(5, tanh.nelement)
        
        let relu = t.relu()
        Assert.AreEqual(5, relu.nelement)
        
        let softplus = t.softplus()
        Assert.AreEqual(5, softplus.nelement)

    [<Test>]
    member _.TestMathematicalFunctionEdgeCases() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float32)
        
        // Test with various special values that might trigger different branches
        let positive = combo.tensor([1.0f; 2.0f; 3.0f])
        let negative = combo.tensor([-1.0f; -2.0f; -3.0f])
        let mixed = combo.tensor([-1.0f; 0.0f; 1.0f])
        
        // Test exp on different value ranges
        let expPos = positive.exp()
        let expMixed = mixed.exp()
        Assert.AreEqual(3, expPos.nelement)
        Assert.AreEqual(3, expMixed.nelement)
        
        // Test log on positive values (negative would be invalid)
        let logPos = positive.log()
        Assert.AreEqual(3, logPos.nelement)
        
        // Test absolute value 
        let absNeg = negative.abs()
        let absMixed = mixed.abs()
        Assert.AreEqual(3, absNeg.nelement)
        Assert.AreEqual(3, absMixed.nelement)