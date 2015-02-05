\name{autocorr}
\alias{autocorr}
\title{Autocorrelation in estimated coefficients.}
\description{
  \code{autocorr} tests the autocorrelation in the coefficients in an SVMMatch object.
  }
\usage{
	autocorr(obj)
	}
\arguments{
  \item{obj}{A fitted SVMMatch object.}
}
\details{Returns the autocorrelation in the posterior means of the coefficients of an SVMMatch object.}
\references{
	Ratkovic, Marc.  2014.  "Balancing within the Margin: Causal Effect Estimation with
Support Vector Machines."  Working paper.	
}
\author{Marc Ratkovic}


\seealso{\link{svmmatch}}

\examples{
\dontrun{
##See svmmatch() for a full implementation
##Load data
	data("LaLonde")
	Data1<-LaLonde
	Data1<-Data1[Data1$exper==0|Data1$treat==1,]
	attach(Data1)	

##Format X matrix
	varnames<-c("age","educ","black","married","nodegr","hisp",
		"re75","re74")
	X<-cbind(Data1[,varnames],Data1$re75==0,Data1$re74==0)
	X<-as.matrix(X)

##Fit model
	set.seed(1)
	m1.param<-svmmatch(treat, X, dv=re78, burnin=100, gibbs=100, thin=5)

##Assess autocorrelation
	ac1<-autocorr(m1.param)
	}
}