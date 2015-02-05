\name{control.overlap}
\alias{control.overlap}
\title{Assessing the number of control observations used in estimating the treatment effect.}
\description{
  \code{control.overlap} returns a diagnostic plot showing the posterior mass over the number of control observations estimated in the common support region between the treatment and control groups.
  }
\usage{
	control.overlap(obj, color=TRUE, label.main="Assessing Control Overlap", 
		label.x="Size of Control Set", label.y="Mass")
	}
\arguments{
  \item{obj}{A fitted SVMMatch object.}
  \item{color}{Whether to plot in color or black and white.  TRUE or FALSE.}
  \item{label.main}{Main title for figure}
  \item{label.x}{X-axis label.}
  \item{label.y}{Y-axis label.}
}
\details{Gives the posterior mass of control observations that fall in the common support region between treated and untreated obsevations.  These are the only observations used in constructing the treatment effect.}
\value{
  \item{counts}{Table containing posterior mass of number untreated observations.}
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

##Assess ignorability:	
	#Assess control overlap--are there enough
	#controls to balance the treated?
		cont1<-control.overlap(m1.param)	

	}
}