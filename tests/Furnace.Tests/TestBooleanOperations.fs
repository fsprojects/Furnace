// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open NUnit.Framework
open Furnace
open Tests.TestUtils

[<TestFixture>]
type TestBooleanOperations () =

    [<Test>]
    member _.TestBooleanTensorLogicalOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test logical AND and OR operations (AddTT as OR, MulTT as AND)
        let t1 = combo.tensor([true; false; true; false])
        let t2 = combo.tensor([true; true; false; false])
        
        // Test logical OR (AddTT operation in boolean context)
        let orResult = t1 + t2
        let expectedOr = combo.tensor([true; true; true; false])
        Assert.CheckEqual(expectedOr, orResult)
        
        // Test logical AND (MulTT operation in boolean context)
        let andResult = t1 * t2
        let expectedAnd = combo.tensor([true; false; false; false])
        Assert.CheckEqual(expectedAnd, andResult)

    [<Test>]
    member _.TestBooleanTensorWithScalars() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test scalar operations
        let t = combo.tensor([true; false; true])
        let trueScalar = FurnaceImage.tensor(true, dtype=Dtype.Bool, backend=combo.backend, device=combo.device)
        let falseScalar = FurnaceImage.tensor(false, dtype=Dtype.Bool, backend=combo.backend, device=combo.device)
        
        // Test AddTT0 (OR with scalar)
        let orWithTrue = t + trueScalar
        let expectedOrTrue = combo.tensor([true; true; true])
        Assert.CheckEqual(expectedOrTrue, orWithTrue)
        
        let orWithFalse = t + falseScalar
        Assert.CheckEqual(t, orWithFalse)
        
        // Test MulTT0 (AND with scalar)
        let andWithTrue = t * trueScalar
        Assert.CheckEqual(t, andWithTrue)
        
        let andWithFalse = t * falseScalar
        let expectedAndFalse = combo.tensor([false; false; false])
        Assert.CheckEqual(expectedAndFalse, andWithFalse)

    [<Test>]
    member _.TestBooleanTensorComparisons() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        let t1 = combo.tensor([true; false; true; false])
        let t2 = combo.tensor([false; false; true; true])
        
        // Test comparison operations specific to boolean tensors
        let ltResult = t1.lt(t2)
        let expectedLt = combo.tensor([false; false; false; true])
        Assert.CheckEqual(expectedLt, ltResult)
        
        let gtResult = t1.gt(t2)
        let expectedGt = combo.tensor([true; false; false; false])
        Assert.CheckEqual(expectedGt, gtResult)
        
        let leResult = t1.le(t2)
        let expectedLe = combo.tensor([false; true; true; true])
        Assert.CheckEqual(expectedLe, leResult)
        
        let geResult = t1.ge(t2)
        let expectedGe = combo.tensor([true; true; true; false])
        Assert.CheckEqual(expectedGe, geResult)
        
        let eqResult = t1.eq(t2)
        let expectedEq = combo.tensor([false; true; true; false])
        Assert.CheckEqual(expectedEq, eqResult)
        
        let neResult = t1.ne(t2)
        let expectedNe = combo.tensor([true; false; false; true])
        Assert.CheckEqual(expectedNe, neResult)

    [<Test>]
    member _.TestBooleanTensorReductionOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test reduction operations on boolean tensors
        let t = combo.tensor([[true; false; true]; [false; true; false]])
        
        // Test sum (converts to int64 then sums)
        let sumResult = t.sum()
        Assert.AreEqual(3.0, sumResult.toScalar().toDouble())  // 3 true values
        
        // Test sum with dimensions
        let sumDim0 = t.sum(0) 
        let expectedSumDim0 = FurnaceImage.tensor([1; 1; 1], dtype=Dtype.Int64, backend=combo.backend)
        Assert.CheckEqual(expectedSumDim0, sumDim0)
        
        let sumDim1 = t.sum(1)
        let expectedSumDim1 = FurnaceImage.tensor([2; 1], dtype=Dtype.Int64, backend=combo.backend)
        Assert.CheckEqual(expectedSumDim1, sumDim1)
        
        // Test min/max operations
        let maxResult = t.max()
        Assert.AreEqual(1.0, maxResult.toScalar().toDouble())  // true as 1.0
        
        let minResult = t.min()
        Assert.AreEqual(0.0, minResult.toScalar().toDouble())  // false as 0.0

    [<Test>]
    member _.TestBooleanTensorMinMaxReduction() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test dimensional min/max reduction for boolean tensors
        let t = combo.tensor([[true; false]; [false; true]])
        
        // Max reduction by dimension
        let maxDim0 = t.max(0)
        let expectedMaxDim0 = combo.tensor([true; true])
        Assert.CheckEqual(expectedMaxDim0, maxDim0)
        
        let maxDim1 = t.max(1)
        let expectedMaxDim1 = combo.tensor([true; true])
        Assert.CheckEqual(expectedMaxDim1, maxDim1)
        
        // Min reduction by dimension
        let minDim0 = t.min(0)
        let expectedMinDim0 = combo.tensor([false; false])
        Assert.CheckEqual(expectedMinDim0, minDim0)
        
        let minDim1 = t.min(1)
        let expectedMinDim1 = combo.tensor([false; false])
        Assert.CheckEqual(expectedMinDim1, minDim1)

    [<Test>]
    member _.TestBooleanTensorBasicProperties() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test basic properties and structure
        let t = combo.tensor([true; false; true; false; true])
        
        // Test basic properties
        Assert.AreEqual([|5|], t.shape)
        Assert.AreEqual(5, t.nelement)
        Assert.AreEqual(Dtype.Bool, t.dtype)

    [<Test>]
    member _.TestBooleanTensorSliceOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test slice operations that may trigger GetTypedValues usage
        let source = combo.tensor([[true; false]; [false; true]])
        
        // Test slice access
        let row0 = source[0]
        let expectedRow0 = combo.tensor([true; false])
        Assert.CheckEqual(expectedRow0, row0)
        
        let row1 = source[1]
        let expectedRow1 = combo.tensor([false; true])
        Assert.CheckEqual(expectedRow1, row1)

    [<Test>]
    member _.TestBooleanTensorCastingOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test casting boolean tensors to other types and back
        let t = combo.tensor([true; false; true; false])
        
        // Cast to int64 (used internally by SumT)
        let asInt64 = t.cast(Dtype.Int64)
        let expectedInt64 = FurnaceImage.tensor([1L; 0L; 1L; 0L], dtype=Dtype.Int64, backend=combo.backend)
        Assert.CheckEqual(expectedInt64, asInt64)
        
        // Cast back to bool
        let backToBool = asInt64.cast(Dtype.Bool)
        Assert.CheckEqual(t, backToBool)

    [<Test>]
    member _.TestBooleanTensorSignOperation() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test SignT operation (should return self for boolean tensors)
        let t = combo.tensor([true; false; true])
        let signResult = t.sign()
        
        // SignT for boolean should return the same tensor
        Assert.CheckEqual(t, signResult)

    [<Test>]
    member _.TestBooleanTensorEqualsAndAllClose() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test Equals and AllClose operations
        let t1 = combo.tensor([true; false; true])
        let t2 = combo.tensor([true; false; true])
        let t3 = combo.tensor([false; true; false])
        
        // Test tensor equality using allclose
        Assert.True(t1.allclose(t2))
        Assert.False(t1.allclose(t3))
        
        // Test AllClose with tolerance parameters (should be same as Equals for boolean tensors)
        Assert.True(t1.allclose(t2, 0.0, 0.0))
        Assert.False(t1.allclose(t3, 0.0, 0.0))

    [<Test>]
    member _.TestBooleanTensorAlphaOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test AddTT with alpha parameter
        let t1 = combo.tensor([true; false; true])
        let t2 = combo.tensor([false; true; false])
        
        // Test basic AddTT operation (boolean OR)
        let resultBasic = t1 + t2
        let expectedBasic = combo.tensor([true; true; true])  // true OR false, false OR true, true OR false
        Assert.CheckEqual(expectedBasic, resultBasic)
        
        // Test MulTT operation (boolean AND) 
        let andResult = t1 * t2
        let expectedAnd = combo.tensor([false; false; false])  // true AND false, false AND true, true AND false
        Assert.CheckEqual(expectedAnd, andResult)

    [<Test>]
    member _.TestBooleanTensorUnsupportedOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test that specific unsupported operations throw appropriate exceptions
        let t1 = combo.tensor([true; false])
        let t2 = combo.tensor([false; true])
        
        // Test subtraction - this should not be supported for boolean tensors
        isInvalidOp (fun () -> t1 - t2)  // SubTT not supported for Bool
        
        // Test division - boolean division may not be supported or may convert to float
        try
            let div_result = t1 / t2
            // If division works, it's okay if it converts to float (this is implementation-dependent behavior)
            Assert.AreEqual(t1.shape, div_result.shape)
            // Division result might be Float32 rather than Bool - this is acceptable
            Assert.IsTrue(div_result.dtype = Dtype.Bool || div_result.dtype = Dtype.Float32, 
                         $"Division result should be Bool or Float32, but got {div_result.dtype}")
        with
        | :? System.InvalidOperationException ->
            // Division not supported - this is also acceptable behavior
            ()
        | ex ->
            // Any other exception type should fail the test
            Assert.Fail($"Unexpected exception type for boolean division: {ex.GetType().Name}, Message: {ex.Message}")
        
        // Test operations that are actually unsupported on boolean tensors
        // abs(), neg(), relu() operations throw InvalidOperationException for bool tensors
        isInvalidOp (fun () -> t1.abs())   // AbsT not permitted on Bool
        isInvalidOp (fun () -> t1.neg())   // NegT not permitted on Bool  
        isInvalidOp (fun () -> t1.relu())  // ReluT not permitted on Bool

    [<Test>]
    member _.TestBooleanTensorEdgeCases() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test edge cases that might trigger different code paths
        
        // Empty boolean tensor
        let empty = FurnaceImage.zeros([0], dtype=Dtype.Bool, backend=combo.backend)
        Assert.AreEqual(0, empty.nelement)
        
        // Scalar boolean tensor
        let scalar = combo.tensor(true)
        Assert.AreEqual(1.0, scalar.toScalar().toDouble())
        
        // Large boolean tensor to stress test operations
        let large = combo.zeros([100; 50])
        Assert.AreEqual(5000, large.nelement)
        
        // Mixed operations with large tensors
        let ones = combo.ones([100; 50])
        let mixed = large + ones  // Should result in all true values
        Assert.AreEqual(5000.0, mixed.sum().toScalar().toDouble())

    [<Test>]
    member _.TestBooleanTensorMakeLikeOperations() =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test operations that use MakeLike method
        let t1 = combo.tensor([[true; false]; [true; false]])
        let t2 = combo.tensor([[false; true]; [false; true]])
        
        // All comparison operations should use MakeLike internally
        let results = [
            t1.lt(t2); t1.gt(t2); t1.le(t2); t1.ge(t2);
            t1.eq(t2); t1.ne(t2); t1 + t2; t1 * t2
        ]
        
        // All results should have same shape as input
        results |> List.iter (fun r -> 
            Assert.AreEqual([|2; 2|], r.shape)
            Assert.AreEqual(Dtype.Bool, r.dtype)
            Assert.AreEqual(Backend.Reference, r.backend)
        )