// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

module Furnace.Shorten

// Functional automatic differentiation API shorthand names
type FurnaceImage with

    /// <summary>TBD</summary>
    static member gvp f x v = FurnaceImage.gradv f x v

    /// <summary>TBD</summary>
    static member g f x = FurnaceImage.grad f x

    /// <summary>TBD</summary>
    static member hvp f x v = FurnaceImage.hessianv f x v

    /// <summary>TBD</summary>
    static member h f x = FurnaceImage.hessian f x

    /// <summary>TBD</summary>
    static member gh f x = FurnaceImage.gradhessian f x

    /// <summary>TBD</summary>
    static member ghvp f x v = FurnaceImage.gradhessianv f x v

    /// <summary>TBD</summary>
    static member jvp f x v = FurnaceImage.jacobianv f x v

    /// <summary>TBD</summary>
    static member vjp f x v = FurnaceImage.jacobianTv f x v

    /// <summary>TBD</summary>
    static member j f x = FurnaceImage.jacobian f x

    /// <summary>TBD</summary>
    static member fgvp f x v = FurnaceImage.fgradv f x v

    /// <summary>TBD</summary>
    static member fg f x = FurnaceImage.fgrad f x

    /// <summary>TBD</summary>
    static member fgh f x = FurnaceImage.fgradhessian f x

    /// <summary>TBD</summary>
    static member fhvp f x v = FurnaceImage.fhessianv f x v

    /// <summary>TBD</summary>
    static member fh f x = FurnaceImage.fhessian f x

    /// <summary>TBD</summary>
    static member fghvp f x v = FurnaceImage.fgradhessianv f x v

    /// <summary>TBD</summary>
    static member fjvp f x v = FurnaceImage.fjacobianv f x v

    /// <summary>TBD</summary>
    static member fvjp f x v = FurnaceImage.fjacobianTv f x v

    /// <summary>TBD</summary>
    static member fj f x = FurnaceImage.fjacobian f x    

