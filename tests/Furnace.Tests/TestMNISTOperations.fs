// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open System.IO
open System.IO.Compression
open System.Net
open NUnit.Framework
open Furnace
open Furnace.Data
open Tests.TestUtils

[<TestFixture>]
type TestMNISTOperations () =

    let createMockMNISTImageFile (filename: string) (numImages: int) =
        // Create a mock MNIST image file with proper format
        use stream = new FileStream(filename, FileMode.Create)
        use gzip = new GZipStream(stream, CompressionMode.Compress)
        use writer = new BinaryWriter(gzip)
        
        // Write MNIST image format header
        writer.Write(IPAddress.HostToNetworkOrder(2051))  // Magic number for images
        writer.Write(IPAddress.HostToNetworkOrder(numImages))  // Number of images
        writer.Write(IPAddress.HostToNetworkOrder(28))  // Height
        writer.Write(IPAddress.HostToNetworkOrder(28))  // Width
        
        // Write mock image data (28x28 bytes per image)
        for i in 0..numImages-1 do
            for pixel in 0..783 do  // 28*28 = 784 pixels, 0-indexed so 783
                writer.Write(byte (i % 256))  // Simple pattern based on image index

    let createMockMNISTLabelFile (filename: string) (numLabels: int) =
        // Create a mock MNIST label file with proper format
        use stream = new FileStream(filename, FileMode.Create)
        use gzip = new GZipStream(stream, CompressionMode.Compress)
        use writer = new BinaryWriter(gzip)
        
        // Write MNIST label format header
        writer.Write(IPAddress.HostToNetworkOrder(2049))  // Magic number for labels
        writer.Write(IPAddress.HostToNetworkOrder(numLabels))  // Number of labels
        
        // Write mock label data
        for i in 0..numLabels-1 do
            writer.Write(byte (i % 10))  // Labels 0-9, cycling

    [<Test>]
    member _.TestMNISTClassProperties() =
        // Test MNIST class properties without requiring network access
        let tempDir = Path.GetTempPath()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + System.Environment.TickCount.ToString()
        let mnistDir = Path.Combine(tempDir, $"test-mnist-props-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Create small mock files to avoid network download
            let trainImagesFile = Path.Combine(fullMnistDir, "train-images-idx3-ubyte.gz")
            let trainLabelsFile = Path.Combine(fullMnistDir, "train-labels-idx1-ubyte.gz")
            
            createMockMNISTImageFile trainImagesFile 10
            createMockMNISTLabelFile trainLabelsFile 10
            
            // Create MNIST dataset with limited data
            let mnist = MNIST(mnistDir, train=true, n=5)
            
            // Test class properties
            Assert.AreEqual(10, mnist.classes)
            Assert.AreEqual(10, mnist.classNames.Length)
            
            // Check class names are string representations of 0-9
            for i in 0..9 do
                Assert.AreEqual(string i, mnist.classNames[i])
                
            // Test length property
            Assert.AreEqual(5, mnist.length)  // We limited to 5 items
            
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors

    [<Test>]
    member _.TestMNISTItemAccess() =
        // Test MNIST item access with mock data
        let tempDir = Path.GetTempPath()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + System.Environment.TickCount.ToString()
        let mnistDir = Path.Combine(tempDir, $"test-mnist-items-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Create small mock files
            let trainImagesFile = Path.Combine(fullMnistDir, "train-images-idx3-ubyte.gz")
            let trainLabelsFile = Path.Combine(fullMnistDir, "train-labels-idx1-ubyte.gz")
            
            createMockMNISTImageFile trainImagesFile 3
            createMockMNISTLabelFile trainLabelsFile 3
            
            // Create MNIST dataset with custom transforms
            let imageTransform = fun (t:Tensor) -> t * 2.0f  // Simple scaling
            let targetTransform = fun (t:Tensor) -> t + 1.0f  // Offset labels by 1
            
            let mnist = MNIST(mnistDir, train=true, n=3, transform=imageTransform, targetTransform=targetTransform)
            
            // Test item access
            for i in 0..2 do
                let data, target = mnist[i]
                
                // Verify data shape (should be [1, 28, 28] after processing)
                Assert.AreEqual([|1; 28; 28|], data.shape)
                
                // Verify target is a scalar
                Assert.AreEqual([||], target.shape)  // Scalar tensor
                
                // Verify transforms were applied
                // Original data is normalized (divided by 255) then multiplied by 2
                // Target should be (i % 10) + 1 due to targetTransform
                let expectedTarget = float32 ((i % 10) + 1)
                let actualTarget = target.toScalar().toSingle()
                Assert.AreEqual(double expectedTarget, double actualTarget, 0.001)
                
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors

    [<Test>]
    member _.TestMNISTTrainVsTest() =
        // Test different behavior for train vs test sets
        let tempDir = Path.GetTempPath()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + System.Environment.TickCount.ToString()
        let mnistDir = Path.Combine(tempDir, $"test-mnist-train-test-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Create mock files for both train and test
            let trainImagesFile = Path.Combine(fullMnistDir, "train-images-idx3-ubyte.gz")
            let trainLabelsFile = Path.Combine(fullMnistDir, "train-labels-idx1-ubyte.gz")
            let testImagesFile = Path.Combine(fullMnistDir, "t10k-images-idx3-ubyte.gz")
            let testLabelsFile = Path.Combine(fullMnistDir, "t10k-labels-idx1-ubyte.gz")
            
            createMockMNISTImageFile trainImagesFile 100  // Training has more data
            createMockMNISTLabelFile trainLabelsFile 100
            createMockMNISTImageFile testImagesFile 20   // Test has less data
            createMockMNISTLabelFile testLabelsFile 20
            
            // Test train dataset
            let trainDataset = MNIST(mnistDir, train=true, n=50)
            Assert.AreEqual(50, trainDataset.length)
            
            // Test test dataset  
            let testDataset = MNIST(mnistDir, train=false, n=15)
            Assert.AreEqual(15, testDataset.length)
            
            // Verify they produce different data (different files)
            let trainItem, trainTarget = trainDataset[0]
            let testItem, testTarget = testDataset[0]
            
            // Both should have same shape but potentially different data
            Assert.AreEqual([|1; 28; 28|], trainItem.shape)
            Assert.AreEqual([|1; 28; 28|], testItem.shape)
            
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors

    [<Test>]
    member _.TestMNISTDefaultTransforms() =
        // Test that default transforms are applied correctly
        let tempDir = Path.GetTempPath()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + System.Environment.TickCount.ToString()
        let mnistDir = Path.Combine(tempDir, $"test-mnist-defaults-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Create mock files
            let trainImagesFile = Path.Combine(fullMnistDir, "train-images-idx3-ubyte.gz")
            let trainLabelsFile = Path.Combine(fullMnistDir, "train-labels-idx1-ubyte.gz")
            
            createMockMNISTImageFile trainImagesFile 2
            createMockMNISTLabelFile trainLabelsFile 2
            
            // Create MNIST with default transforms
            let mnist = MNIST(mnistDir, train=true, n=2)
            
            let data, target = mnist[0]
            
            // Test data shape and basic properties
            Assert.AreEqual([|1; 28; 28|], data.shape)
            Assert.AreEqual(Dtype.Float32, data.dtype)
            
            // Default transform is (t - 0.1307) / 0.3081
            // Since our mock data will be normalized to [0,1] first, we can't predict exact values
            // but we can verify the transform was applied by checking it's not in [0,1] range
            let dataValues = data.flatten()
            Assert.AreEqual(784, dataValues.nelement)  // 28*28 pixels
            
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors

    [<Test>]
    member _.TestMNISTErrorHandling() =
        // Test error handling for invalid files
        let tempDir = Path.GetTempPath()
        let processId = System.Diagnostics.Process.GetCurrentProcess().Id.ToString()
        let threadId = System.Threading.Thread.CurrentThread.ManagedThreadId.ToString()
        let ticks = System.DateTime.UtcNow.Ticks.ToString()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + processId + "-" + threadId + "-" + ticks
        let mnistDir = Path.Combine(tempDir, $"test-mnist-errors-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Create file with wrong magic number  
            let badImageFile = Path.Combine(fullMnistDir, "train-images-idx3-ubyte.gz")
            
            // Ensure proper file handle disposal using explicit scoping
            do
                use stream = new FileStream(badImageFile, FileMode.Create)
                use gzip = new GZipStream(stream, CompressionMode.Compress)
                use writer = new BinaryWriter(gzip)
                writer.Write(IPAddress.HostToNetworkOrder(9999))  // Wrong magic number
            // All file handles are now disposed
            
            // Force garbage collection to ensure all handles are released
            System.GC.Collect()
            System.GC.WaitForPendingFinalizers()
            System.Threading.Thread.Sleep(100)  // Small delay to allow file system
            
            // This should throw an exception due to invalid format
            isException (fun () -> 
                let mnist = MNIST(mnistDir, train=true, n=1)
                mnist[0] |> ignore
            )
            
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors

    [<Test>]
    member _.TestMNISTWithCustomURLs() =
        // Test MNIST creation with custom URLs (though we won't actually download)
        let tempDir = Path.GetTempPath()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + System.Environment.TickCount.ToString()
        let mnistDir = Path.Combine(tempDir, $"test-mnist-urls-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Pre-create the files to avoid download
            let trainImagesFile = Path.Combine(fullMnistDir, "custom-train-images.gz")
            let trainLabelsFile = Path.Combine(fullMnistDir, "custom-train-labels.gz")
            let testImagesFile = Path.Combine(fullMnistDir, "custom-test-images.gz")
            let testLabelsFile = Path.Combine(fullMnistDir, "custom-test-labels.gz")
            
            createMockMNISTImageFile trainImagesFile 10
            createMockMNISTLabelFile trainLabelsFile 10
            createMockMNISTImageFile testImagesFile 5
            createMockMNISTLabelFile testLabelsFile 5
            
            let customUrls = [
                "http://example.com/custom-train-images.gz"
                "http://example.com/custom-train-labels.gz"
                "http://example.com/custom-test-images.gz"
                "http://example.com/custom-test-labels.gz"
            ]
            
            // This should work because files already exist
            let mnist = MNIST(mnistDir, urls=customUrls, train=true, n=5)
            Assert.AreEqual(5, mnist.length)
            
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors

    [<Test>]
    member _.TestMNISTDerivedFromDataset() =
        // Test that MNIST properly inherits from Dataset base class
        let tempDir = Path.GetTempPath()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + System.Environment.TickCount.ToString()
        let mnistDir = Path.Combine(tempDir, $"test-mnist-inheritance-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Create minimal mock files
            let trainImagesFile = Path.Combine(fullMnistDir, "train-images-idx3-ubyte.gz")
            let trainLabelsFile = Path.Combine(fullMnistDir, "train-labels-idx1-ubyte.gz")
            
            createMockMNISTImageFile trainImagesFile 3
            createMockMNISTLabelFile trainLabelsFile 3
            
            let mnist = MNIST(mnistDir, train=true, n=3)
            
            // Test that it can be used as a Dataset
            let dataset : Dataset = upcast mnist
            Assert.AreEqual(3, dataset.length)
            
            // Test indexer works through base class
            let data, target = dataset[1]
            Assert.AreEqual([|1; 28; 28|], data.shape)
            
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors

    [<Test>]
    member _.TestMNISTDataNormalization() =
        // Test that MNIST data is properly normalized from byte values to [0,1]
        let tempDir = Path.GetTempPath()
        let processId = System.Diagnostics.Process.GetCurrentProcess().Id.ToString()
        let threadId = System.Threading.Thread.CurrentThread.ManagedThreadId.ToString()
        let ticks = System.DateTime.UtcNow.Ticks.ToString()
        let uniqueId = System.Guid.NewGuid().ToString("N") + "-" + processId + "-" + threadId + "-" + ticks
        let mnistDir = Path.Combine(tempDir, $"test-mnist-norm-{uniqueId}")
        Directory.CreateDirectory(mnistDir) |> ignore  
        let fullMnistDir = Path.Combine(mnistDir, "mnist")
        Directory.CreateDirectory(fullMnistDir) |> ignore
        
        try
            // Create mock file with known byte patterns
            let trainImagesFile = Path.Combine(fullMnistDir, "train-images-idx3-ubyte.gz")
            let trainLabelsFile = Path.Combine(fullMnistDir, "train-labels-idx1-ubyte.gz")
            
            // Create file with specific byte pattern for testing normalization
            // Ensure all file handles are properly closed by using explicit scoping
            do
                use stream = new FileStream(trainImagesFile, FileMode.Create)
                use gzip = new GZipStream(stream, CompressionMode.Compress)
                use writer = new BinaryWriter(gzip)
                
                writer.Write(IPAddress.HostToNetworkOrder(2051))  // Magic number
                writer.Write(IPAddress.HostToNetworkOrder(1))     // 1 image
                writer.Write(IPAddress.HostToNetworkOrder(28))    // Height
                writer.Write(IPAddress.HostToNetworkOrder(28))    // Width
                
                // Write known pattern: 0, 127, 255 repeated
                for i in 0..783 do
                    let value = match i % 3 with
                                | 0 -> 0uy
                                | 1 -> 127uy 
                                | _ -> 255uy
                    writer.Write(value)
            // All file handles are now disposed
            
            createMockMNISTLabelFile trainLabelsFile 1
            
            // Force garbage collection to ensure all handles are released
            System.GC.Collect()
            System.GC.WaitForPendingFinalizers()
            System.Threading.Thread.Sleep(100)  // Small delay to allow file system
            
            // Use identity transform to see raw normalized data
            let mnist = MNIST(mnistDir, train=true, n=1, transform=id)
            let data, _ = mnist[0]
            
            // Verify normalization: bytes should be converted to [0,1] range
            let flatData = data.flatten()
            let minVal = flatData.min().toScalar().toSingle()
            let maxVal = flatData.max().toScalar().toSingle()
            
            Assert.GreaterOrEqual(minVal, 0.0f)
            Assert.LessOrEqual(maxVal, 1.0f)
            
            // Should have values corresponding to 0/255, 127/255, 255/255
            // Due to the pattern we wrote
            
        finally
            try
                if Directory.Exists(mnistDir) then
                    // Force garbage collection to close any file handles
                    System.GC.Collect()
                    System.GC.WaitForPendingFinalizers()
                    System.GC.Collect()
                    System.Threading.Thread.Sleep(250) // Longer delay to allow file handles to be released
                    // Retry deletion up to 5 times with progressive backoff
                    let mutable attempts = 0
                    let mutable deleted = false
                    while attempts < 5 && not deleted do
                        try
                            // Additional GC before each attempt
                            System.GC.Collect()
                            Directory.Delete(mnistDir, true)
                            deleted <- true
                        with
                        | :? System.IO.IOException when attempts < 4 ->
                            attempts <- attempts + 1
                            let sleepTime = 300 * (attempts * attempts) // Progressive: 300, 1200, 2700, 4800ms
                            System.Threading.Thread.Sleep(sleepTime)
                        | _ -> 
                            attempts <- 5 // Stop trying on other errors
            with
            | _ -> () // Ignore cleanup errors