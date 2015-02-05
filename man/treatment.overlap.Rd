\name{treatment.overlap}
\alias{treatment.overlap}
\title{Exploring hard-to-match treated observations.}
\description{
  \code{treatment overlap} characterizes treated observations which have no natural matches in the data because they fall outside the common support region.
  }
\usage{
	treatment.overlap(obj, color=TRUE, thresh=.95)	
	}
\arguments{
  \item{obj}{A fitted SVMMatch object.}
  \item{color}{Whether to plot in color or black and white.  TRUE or FALSE.}
  \item{thresh}{Proportion of time a treated observation must fall outside the common support region in order to be considered unmatched.}
}
\details{SVMMatch estimates a region of common support between treated and untreated observations.  This function identifies and characterizes the treated observations that regularly fall outside the common support region.  Having too many of these observations makes estimation of the treatment effect casts doubt on whether the average treatment effect is identified in the data.}
\value{
  \item{no.overlap}{A binary vector the length of the number of treated observations taking a value of one if the observation falls outside the region of common support at some rate above the threshold.}
  \item{logit}{A logistic regression using the original covariates in the SVMMatch object to predict whether an observation falls outside the common support region.}
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

	#Assess treatment overlap--are there treated 
	#values that can't be matched?
		treat1<-treatment.overlap(m1.param)

	}
}