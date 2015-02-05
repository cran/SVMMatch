\name{sensitivity}
\alias{sensitivity}
\title{Sensitivity analysis for SVMMatch.}
\description{
  \code{sensitivity} assesses the sensitivity of an effect estimate to an omitted confounder.
  }
\usage{
	sensitivity(obj, seq.eval=seq(-1,1,.1), quant.eval=c(0.025,0.5,0.975), 
		color=TRUE, legend.pos="topleft", label.main="Sensitivity Analysis",
		label.x="Sensitivity Parameter",  label.y="Outcome")	
	}
\arguments{
  \item{obj}{A fitted SVMMatch object.}
  \item{seq.eval}{Values at which to set the omitted confounder, in the range [-1, +1].}
  \item{quant.eval}{Values at which to plot the posterior density as a function of the omitted confounder.  By default, a solid line is drawn through the posterior medians, with dashed lines at the 2.5th and 97.5th percentiles.} 
  \item{color}{Whether to plot in color or black and white.  TRUE or FALSE.}
  \item{legend.pos}{Where to place the margin.  See the help file for \code{legend}.}
  \item{label.main}{Main title for figure}
  \item{label.x}{X-axis label.}
  \item{label.y}{Y-axis label.}
}
\details{Conducts a sensitivity analysis using an SVMMatch object.  An unoberseved parameter, u, that predicts the treatment assignment is introduced and varied between -1 and 1.  For each value of u, balancing weights are constructed and the posterior density of the effect estimate recalcluated, with u=0 returning the results from the original fit.  The figure gives the researcher a sense as to how sensitive the effect estimate is to omitted confounders.}
\value{
  \item{sens.mat}{A matrix of the posterior estimates as a function of the unobserved confounder.}
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

##Sensitivity analysis (Takes a little longer)
	sens1<-sensitivity(m1.param)

	}
}