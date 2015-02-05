\name{SVMMatch-package}
\alias{SVMMatch-package}
\alias{SVMMatch}
\docType{package}
\title{
Title: Causal effect estimation and diagnostics with support vector machines.
}
\description{
SVMMatch identifies the region of common support between a set of treated and control units in observational data.  Using the observations in this region, a set of balancing weights and a treatment effect are estimated. The method, described in Ratkovic (2014), adapts the support vector machine technology in order to estimate these balancing weights, using a Bayesian implementation so as to give uncertainty effects both in treatment assignment and effect estimation.
}
\details{
\tabular{ll}{
Package: \tab SVMMatch\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2015-02-05\cr
License: \tab GPL (>= 2)\cr
}
The method implements the matching algorithm through the main function, svmmatch.

A series of diagnostics are implemented.  The function balance() assesses the posterior density of covariate imbalance; effect() returns the posterior estimate of the treatment effect; sensitivity() assesses the effect estimate's sensitivity to unobserved confounders; control.overlap() returns the posterior density of number of control observations returned in matching; and treatment.overlap() examines difficult-to-match treated observations.
}
\author{
Marc Ratkovic\cr
Maintainer: Marc Ratkovic <ratkovic@princeton.edu>
}
\references{
Ratkovic, Marc.  2014.  "Balancing within the Margin: Causal Effect Estimation with
Support Vector Machines."  Working paper.
}
\keyword{ package }


