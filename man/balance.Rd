\name{balance}
\alias{balance}
\title{Assessing balance when using SVMMatch.}
\description{
  \code{balance} returns a diagnostic plot of covariate balance as well as an object containing the posterior estimate of covariate imbalance.}
\usage{
	balance(treat, X, obj, plot.it=TRUE, sd.plot=.2, color=TRUE)
	}
\arguments{
  \item{treat}{A binary vector indicating treatment status.}
  \item{X}{A covariate matrix.}
  \item{obj}{A fitted SVMMatch object.}
  \item{plot.it}{Whether to return a figure illustrating balancing in the raw and balanced data.  TRUE or FALSE.}
  \item{sd.plot}{Number of standard deviations at which to include dashed vertical lines in the figure.}
  \item{color}{Whether to plot in color or black and white.  TRUE or FALSE.}
}
\details{Assess the covariate balance using weights from an SVMMatch object.}
\value{
  \item{balance}{Posterior density of imbalance, by covariate.}
 }
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

##Assess balance
	bal1<-balance(treat, X, m1.param)

	}
}