// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

module Furnace.Numerical

// Functional numerical differentiation API
type FurnaceImage with

    /// <summary>TBD</summary>
    static member numdiff (epsilon:float) (f:Tensor->Tensor) (x:Tensor) = 
        if x.dim <> 0 then failwithf "f must be a function of a scalar"
        ((f (x + epsilon)) - (f (x - epsilon))) / (2.*epsilon)

    /// <summary>TBD</summary>
    static member numfdiff epsilon f x = f x, FurnaceImage.numdiff epsilon f x

    /// <summary>TBD</summary>
    static member numfdiff2 (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        if x.dim <> 0 then failwithf "f must be a function of a scalar"
        let fx = f x
        fx, ((f (x + epsilon)) - 2. * fx + (f (x - epsilon))) / (epsilon * epsilon)

    /// <summary>TBD</summary>
    static member numdiff2 epsilon f x = FurnaceImage.numfdiff2 epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numjacobianv (epsilon:float) (f:Tensor->Tensor) (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let veps = v * epsilon
        let fxa, fxb = f (x+veps), f (x-veps)
        if x.dim <> 1 || fxa.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fxa.shape
        (fxa - fxb) / (2.*epsilon)

    /// <summary>TBD</summary>
    static member numfjacobianv epsilon f x v = f x, FurnaceImage.numjacobianv epsilon f x v

    /// <summary>TBD</summary>
    static member numfjacobian (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        let fx = f x
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        let j = fx.expand([x.nelement; fx.nelement])
        let jj = FurnaceImage.stack(Array.init x.nelement (fun i -> f (x + FurnaceImage.onehot(x.nelement, i)*epsilon)))
        fx, (jj - j).transpose() / epsilon

    /// <summary>TBD</summary>
    static member numjacobian epsilon f x = FurnaceImage.numfjacobian epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numgradv (epsilon:float) (f:Tensor->Tensor) (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let veps = v * epsilon
        let fxa, fxb = f (x + veps), f (x - veps)
        if x.dim <> 1 || fxa.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fxa.shape
        (fxa - fxb) / (2.*epsilon)

    /// <summary>TBD</summary>
    static member numfgradv epsilon f x v = f x, FurnaceImage.numgradv epsilon f x v

    /// <summary>TBD</summary>
    static member numfgrad (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        if x.dim = 0 then
            FurnaceImage.numfdiff epsilon f x
        else
            let fx = f x
            if x.dim > 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector or scalar, encountered f:%A->%A" x.shape fx.shape
            let gg = FurnaceImage.stack(Array.init x.nelement (fun i -> let h = FurnaceImage.onehot(x.nelement, i)*epsilon in f (x + h) - f (x - h)))
            fx, gg/(2.*epsilon)

    /// <summary>TBD</summary>
    static member numgrad epsilon f x = FurnaceImage.numfgrad epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfgradhessian (epsilon:float) (f:Tensor->Tensor) (x:Tensor) =
        let fx, g = FurnaceImage.numfgrad epsilon f x
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        let h = g.expand([x.nelement; x.nelement])
        let hh = FurnaceImage.stack(Array.init x.nelement (fun i -> FurnaceImage.numgrad epsilon f (x + FurnaceImage.onehot(x.nelement, i)*epsilon)))
        fx, g, (hh - h) / epsilon

    /// <summary>TBD</summary>
    static member numgradhessian epsilon f x = let _, g, h = FurnaceImage.numfgradhessian epsilon f x in g, h

    /// <summary>TBD</summary>
    static member numfhessian epsilon f x = let fx, _, h = FurnaceImage.numfgradhessian epsilon f x in fx, h

    /// <summary>TBD</summary>
    static member numhessian epsilon f x = FurnaceImage.numfhessian epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfhessianv (epsilon:float) (f:Tensor->Tensor) (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let veps = v*epsilon
        let fx, g = FurnaceImage.numfgrad epsilon f x
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        let gg = FurnaceImage.numgrad epsilon f (x + veps)
        fx, (gg-g)/epsilon

    /// <summary>TBD</summary>
    static member numhessianv epsilon f x v = FurnaceImage.numfhessianv epsilon f x v |> snd

    /// <summary>TBD</summary>
    static member numflaplacian epsilon f x =
        let fx, h = FurnaceImage.numfhessian epsilon f x
        fx, h.trace()

    /// <summary>TBD</summary>
    static member numlaplacian epsilon f x = FurnaceImage.numflaplacian epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfcurl epsilon f x =
        let fx, j = FurnaceImage.numfjacobian epsilon f x
        if j.shape <> [|3; 3|] then failwithf "f must be a function with a three-by-three Jacobian"
        fx, FurnaceImage.stack([j[2, 1] - j[1, 2]; j[0, 2] - j[2, 0]; j[1, 0] - j[0, 1]])

    /// <summary>TBD</summary>
    static member numcurl epsilon f x = FurnaceImage.numfcurl epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfdivergence epsilon f x =
        let fx, j = FurnaceImage.numfjacobian epsilon f x
        if j.shape[0] <> j.shape[1] then failwithf "f must have a square Jacobian"
        fx, j.trace()

    /// <summary>TBD</summary>
    static member numdivergence epsilon f x = FurnaceImage.numfdivergence epsilon f x |> snd

    /// <summary>TBD</summary>
    static member numfcurldivergence epsilon f x =
        let fx, j = FurnaceImage.numfjacobian epsilon f x
        if j.shape <> [|3; 3|] then failwithf "f must be a function with a three-by-three Jacobian"
        fx, FurnaceImage.stack([j[2, 1] - j[1, 2]; j[0, 2] - j[2, 0]; j[1, 0] - j[0, 1]]), j.trace()

    /// <summary>TBD</summary>
    static member numcurldivergence epsilon f x = let _, c, d = FurnaceImage.numfcurldivergence epsilon f x in c, d


module Shorten =
    // Functional numerical differentiation API shorthand names
    type FurnaceImage with

        /// <summary>TBD</summary>
        static member numgvp f x v = FurnaceImage.numgradv f x v

        /// <summary>TBD</summary>
        static member numg f x = FurnaceImage.numgrad f x

        /// <summary>TBD</summary>
        static member numhvp f x v = FurnaceImage.numhessianv f x v

        /// <summary>TBD</summary>
        static member numh f x = FurnaceImage.numhessian f x

        /// <summary>TBD</summary>
        static member numgh f x = FurnaceImage.numgradhessian f x

        /// <summary>TBD</summary>
        static member numjvp f x v = FurnaceImage.numjacobianv f x v

        /// <summary>TBD</summary>
        static member numj f x = FurnaceImage.numjacobian f x

        /// <summary>TBD</summary>
        static member numfgvp f x v = FurnaceImage.numfgradv f x v

        /// <summary>TBD</summary>
        static member numfg f x = FurnaceImage.numfgrad f x

        /// <summary>TBD</summary>
        static member numfhvp f x v = FurnaceImage.numfhessianv f x v

        /// <summary>TBD</summary>
        static member numfh f x = FurnaceImage.numfhessian f x

        /// <summary>TBD</summary>
        static member numfgh f x = FurnaceImage.numfgradhessian f x

        /// <summary>TBD</summary>
        static member numfjvp f x v = FurnaceImage.numfjacobianv f x v

        /// <summary>TBD</summary>
        static member numfj f x = FurnaceImage.numfjacobian f x    