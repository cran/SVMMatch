\name{bayesmatch_cpp}
\alias{bayesmatch_cpp}
\title{Rcpp implementation for Bayesian SVM.}
\description{
  \code{bayesmatch_cpp} is the Rcpp object that fits the Bayesian SVM.  Primarily for internal use.
  }
\usage{
	bayesmatch_cpp(X0, boldX0, lambda0, treat0, total_gibbs0, thin0,
                 param0, dv0, nu0, burnin0)

	}
\arguments{
  \item{X0}{Scaled covariate matrix.}
  \item{boldX0}{Scaled and transformed covariate matrix.}
  \item{lambda0}{Vector of initial weights for each observation} 
  \item{treat0}{Binary treatment vector.}
  \item{total_gibbs0}{Number of posterior draws to save.}
  \item{thin0}{Number of draws to thin.}
  \item{param0}{No longer used.  Set to 0.}
  \item{dv0}{Outcome variable.}
  \item{nu0}{Initial estimate for prior weight.}
  \item{burnin0}{Number of posterior draws discarded.}
}
\details{Internal function.}
\references{
	Ratkovic, Marc.  2014.  "Balancing within the Margin: Causal Effect Estimation with
Support Vector Machines."  Working paper.	
}
\author{Marc Ratkovic}


\seealso{\link{svmmatch}}

\examples{
## See svmmatch() for implementation.
}