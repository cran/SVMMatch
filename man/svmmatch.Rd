\name{svmmatch}
\alias{svmmatch}
\title{SVMMatch for Causal Effect Estimation}
\description{
  \code{svmmatch} estimates balancing weights in the presence of non-random treatment assignment. The method allows the researcher to assess the two crucial ignorability assumptions: a test of sensitivity to omitted confounders and a test of common support between treated and control.  A fully Bayesian implementation allows characterization of the posterior of the treatment effect.  The method is implemented in C++ through Rcpp, greatly speeding calculation time.}
\usage{
	  svmmatch(treat, X, burnin = 100, gibbs = 200, thin = 2, dv = NULL)
}
\arguments{
  \item{treat}{A binary vector indicating treatment status.}
  \item{X}{A covariate matrix.}
  \item{burnin}{Number of burnin draws for the Gibbs sampler.}
  \item{gibbs}{Number of posterior draws to be saved.}
  \item{thin}{Number of chains when thinning.  If burnin = 10, gibbs = 20, and thin=30,then 500 = 10 x 30 + 20 x 30 samples will be drawn, but only 20 returned.}
  \item{dv}{A dependent variable of interest.  Optional.}
}
\details{Fits SVMMatch to the data, returning balancing weights and a posterior density of the estimated effect.}
\value{
  \item{effect}{Posterior density of treatment effect on dv.}
  \item{beta}{Matrix of posterior draws for coefficients.}
  \item{margin}{Binary vector for posterior estimate of marginal observations.}
  \item{bal.wts}{Balancing weights.}
  \item{X.scale}{Scaled covariate matrix.  Used internally.}
  \item{X.orig}{Original matrix of covariates.}
  \item{treat}{Treatment vector.}
  \item{dv}{The dependent variable.}
 }
\references{
	Ratkovic, Marc.  2014.  "Balancing within the Margin: Causal Effect Estimation with
Support Vector Machines."  Working paper.	
}
\author{Marc Ratkovic}


\seealso{\link{balance}, \link{effect}, \link{sensitivity}, \link{control.overlap}, \link{treatment.overlap}, \link{LaLonde}}

\examples{

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

##Summarize treatment effect
	effect1<-effect(m1.param)

\dontrun{
##Sensitivity analysis (Takes a little longer)
	sens1<-sensitivity(m1.param)
	}

##Assess ignorability:	
	#Assess control overlap--are there enough
	#controls to balance the treated?
		cont1<-control.overlap(m1.param)	

	#Assess treatment overlap--are there treated 
	#values that can't be matched?
		treat1<-treatment.overlap(m1.param)

##Assess autocorrelation
	ac1<-autocorr(m1.param)

}