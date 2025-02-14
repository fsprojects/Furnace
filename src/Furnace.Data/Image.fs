namespace Furnace

open SkiaSharp

[<AutoOpen>]
module ImageUtil =
    /// Saves the given pixel array to a file and optionally resizes it in the process. Supports .png format.
    let saveImage (pixels: float32[,,]) (fileName: string) (resize: option<int * int>) : unit =
        let c, h, w = pixels.GetLength 0, pixels.GetLength 1, pixels.GetLength 2
        
        use bitmap = new SKBitmap(w, h, SKColorType.Bgra8888, SKAlphaType.Premul)
        use surface = SKSurface.Create(bitmap.Info)
        use canvas = surface.Canvas
        
        for y in 0 .. h - 1 do
            for x in 0 .. w - 1 do
                let r, g, b =
                    if c = 1 then
                        let gray = int (pixels.[0, y, x] * 255.0f)
                        gray, gray, gray
                    else
                        let r = int (pixels.[0, y, x] * 255.0f)
                        let g = int (pixels.[1, y, x] * 255.0f)
                        let b = int (pixels.[2, y, x] * 255.0f)
                        r, g, b
                bitmap.SetPixel(x, y, SKColor(byte r, byte g, byte b))

        use resized = 
            match resize with
            | Some (width, height) ->
                let info = new SKImageInfo(width, height)
                use resizedBitmap = bitmap.Resize(info, SKFilterQuality.High)
                resizedBitmap.Encode(SKEncodedImageFormat.Png, 100)
            | None ->
                bitmap.Encode(SKEncodedImageFormat.Png, 100)

        use stream = System.IO.File.OpenWrite(fileName)
        resized.SaveTo(stream)

    /// Loads a pixel array from a file and optionally resizes it in the process.
    let loadImage (fileName: string) (resize: option<int * int>) : float32[,,] =
        use stream = System.IO.File.OpenRead(fileName)
        use codec = SKCodec.Create(stream)
        use bitmap = SKBitmap.Decode(codec)
        
        let bitmap =
            match resize with
            | Some (width, height) ->
                let info = new SKImageInfo(width, height)
                bitmap.Resize(info, SKFilterQuality.High)
            | None -> bitmap

        let w, h = bitmap.Width, bitmap.Height
        let pixels = Array3D.create 3 h w 0.0f

        for y in 0 .. h - 1 do
            for x in 0 .. w - 1 do
                let color = bitmap.GetPixel(x, y)
                pixels.[0, y, x] <- float32 color.Red / 255.0f
                pixels.[1, y, x] <- float32 color.Green / 255.0f
                pixels.[2, y, x] <- float32 color.Blue / 255.0f

        pixels


[<AutoOpen>]
module ImageExtensions =
    type Tensor with
        /// <summary>Save tensor to an image file using png or jpg format</summary>
        member t.saveImage(fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?resize:int*int, ?gridCols:int) =
            let pixels:Tensor = t.move(Device.CPU).toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)
            saveImage (pixels.float32().toArray() :?> float32[,,]) fileName resize

        /// <summary>Load an image file and return it as a tensor</summary>
        static member loadImage(fileName:string, ?normalize:bool, ?resize:int*int, ?device: Device, ?dtype: Dtype, ?backend: Backend) =
            let normalize = defaultArg normalize false
            let pixels = loadImage fileName resize
            let pixels:Tensor = Tensor.create(pixels, ?device=device, ?dtype=dtype, ?backend=backend)
            if normalize then pixels.normalize() else pixels


    type FurnaceImage with
        /// <summary>Load an image file as a tensor.</summary>
        /// <param name="fileName">The file name of the image to load.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1).</param>
        /// <param name="resize">An optional new size for the image.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member loadImage(fileName:string, ?normalize:bool, ?resize:int*int, ?device: Device, ?dtype: Dtype, ?backend: Backend) =
            Tensor.loadImage(fileName=fileName, ?normalize=normalize, ?resize=resize, ?device=device, ?dtype=dtype, ?backend=backend)

        /// <summary>Save a given Tensor into an image file.</summary>
        /// <remarks>If the input tensor has 4 dimensions, then make a single image grid.</remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="fileName">The name of the file to save to.</param>
        /// <param name="pixelMin">The minimum pixel value.</param>
        /// <param name="pixelMax">The maximum pixel value.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1), by the min and max values specified by range.</param>
        /// <param name="resize">An optional new size for the image.</param>
        /// <param name="gridCols">Number of columns of images in the grid.</param>
        static member saveImage(input:Tensor, fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?resize:int*int, ?gridCols:int) =
            input.saveImage(fileName=fileName, ?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?resize=resize, ?gridCols=gridCols)
