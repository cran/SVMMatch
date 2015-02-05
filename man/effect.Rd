\name{effect}
\alias{effect}
\title{Posterior density of the treatment effect estimate from an SVMMatch object.}
\description{
  \code{effect} characterizes the posterior density of the treatment effect, as estimated through SVMMatch. 
  }
\usage{
	effect(obj, color=TRUE, quant=c(0.025,0.975), legend.pos="topleft",
		label.main="Posterior Density of Effect Estimate", label.x="Outcome",
		label.y="Density")
		}
\arguments{
  \item{obj}{A fitted SVMMatch object.}
  \item{color}{Whether to plot in color or black and white.  TRUE or FALSE.}
  \item{quant}{Quantiles of the effect to be marked on the figure.}
  \item{legend.pos}{Where to place the margin.  See the help file for \code{legend}.}
  \item{label.main}{Main title for figure}
  \item{label.x}{X-axis label.}
  \item{label.y}{Y-axis label.}
}
\details{Characterizes and plots the posterior density of the treatment effect, given an SVMMatch object.  At each posterior draw, a set of balancing weights are constructed, as described in Ratkovic (2014).  A treatment effect is estimated for each posterior draw of balancing weights, and the posterior density returned as a figure.}
\value{
  \item{balance}{Posterior density of imbalance, by covariate.}
 }
\references{
	Ratkovic, Marc.  2014.  "Balancing within the Margin: Causal Effect Estimation with
Support Vector Machines."  Working paper.	
}
\author{Marc Ratkovic}


\seealso{\link{svmmatch}, \link{legend}}

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

##Summarize treatment effect
	effect1<-effect(m1.param)

	}
}