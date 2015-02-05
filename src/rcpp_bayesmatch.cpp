
#include <RcppArmadillo.h>
#define ARMA_64BIT_WORD

using namespace Rcpp;

// [[Rcpp::export]]
List bayesmatch_cpp(
SEXP X0, SEXP boldX0, SEXP lambda0,
SEXP treat0, SEXP total_gibbs0, SEXP thin0,
SEXP param0, SEXP dv0, SEXP nu0, SEXP burnin0
){

//Declare inputs
arma::mat X = Rcpp::as< arma::mat >(X0);
arma::mat boldX = Rcpp::as< arma::mat >(boldX0);
arma::mat XtX=arma::strans(boldX)*boldX;
arma::mat XtX_inv=XtX;
Rcpp::NumericVector lambda=lambda0;
Rcpp::NumericVector treat=treat0;
Rcpp::NumericVector y = 2*treat-1;
Rcpp::NumericVector ranscalar = 0*treat;
Rcpp::NumericVector ranunif = 0*treat;

int lambda_length=lambda.size();
int colsX = boldX.n_cols;
int rowsX = boldX.n_rows;
int burnin = Rcpp::as<int>(burnin0);
int total_gibbs = Rcpp::as<int>(total_gibbs0);
int thin = Rcpp::as<int>(thin0);
int param = Rcpp::as<int>(param0);

arma::mat b_next=arma::zeros<arma::mat>(colsX,1);
arma::mat b_last=arma::zeros<arma::mat>(colsX,1);
arma::mat y_synth=arma::zeros<arma::mat>(rowsX,1);
arma::mat margvec=arma::zeros<arma::mat>(rowsX,rowsX);
arma::mat temp_coef=arma::zeros<arma::mat>(2,1);
arma::mat dv =  Rcpp::as< arma::mat >(dv0);
arma::mat effect = arma::zeros<arma::mat>(total_gibbs,1);
arma::mat b_run = arma::zeros<arma::mat>(total_gibbs,colsX);
arma::mat b_samp =  arma::zeros<arma::mat>(colsX,1);
arma::mat b_ran =  arma::zeros<arma::mat>(colsX,20);
arma::mat margrun = arma::zeros<arma::mat>(total_gibbs,rowsX);
arma::mat wtsrun = arma::zeros<arma::mat>(total_gibbs,rowsX);
arma::mat wtsvec = arma::zeros<arma::mat>(rowsX,1);
arma::mat XtX_inv_chol = arma::zeros<arma::mat>(colsX,colsX);
arma::mat sort_temp = arma::zeros<arma::mat>(colsX,1);

double nu = Rcpp::as<double>(nu0);
double nu_invg = Rcpp::as<double>(nu0);
double mu_invg = Rcpp::as<double>(nu0);
double lambda_invg = Rcpp::as<double>(nu0);
double x_invg = Rcpp::as<double>(nu0);
double y_invg = Rcpp::as<double>(nu0);
double part1 = Rcpp::as<double>(nu0);
double part2 = Rcpp::as<double>(nu0);
double part3 = Rcpp::as<double>(nu0);
double sumb = Rcpp::as<double>(nu0);

arma::mat diaglambda = arma::eye<arma::mat>(lambda_length,lambda_length);
arma::mat diagridge = arma::eye<arma::mat>(colsX,colsX);
arma::mat fits = arma::zeros<arma::mat>(rowsX,1);
arma::mat X_uncond = arma::ones<arma::mat>(rowsX,2);

int posfit = Rcpp::as<int>(nu0); 
int negfit = Rcpp::as<int>(nu0);

for(int i_temp=0; i_temp < rowsX; i_temp++) {X_uncond(i_temp,1) = treat(i_temp);}


//Start loop here
for(int i_outer=0; i_outer<burnin+total_gibbs; i_outer++){
for(int i_thin=0; i_thin<thin; i_thin++){

//Update X.t()*lambda*X, and synthetic y
for(int i_lambda=0; i_lambda<rowsX; i_lambda++) {
if(treat(i_lambda)==0) {
diaglambda(i_lambda,i_lambda) = 1/lambda(i_lambda);
y_synth(i_lambda,0) = (1+lambda(i_lambda));
} else{
diaglambda(i_lambda,i_lambda) = 0;
y_synth(i_lambda,0) = 0;
}
}

//Calcualte mean and variance of b
XtX_inv = arma::pinv(arma::strans(boldX)*arma::diagmat(diaglambda)*boldX+pow(nu,-2) *diagridge);
b_next = XtX_inv*arma::strans(boldX)*arma::diagmat(diaglambda)*y_synth;

//Sample b
int ncols_ran = XtX_inv.n_cols;
b_last = b_samp;

XtX_inv_chol = arma::chol(XtX_inv);

//Try selecting
//arma::mat ranvec = arma::randn(ncols_ran,20);
//b_ran = XtX_inv_chol*ranvec;

//for(int i_brow=0; i_brow<ncols_ran; i_brow++){
//for(int i_bcol=0; i_bcol<20; i_bcol++){
//b_ran(i_brow,i_bcol) = b_ran(i_brow,i_bcol) + b_next(i_brow,0);
//}
//}

//for(int i_brow=0; i_brow<ncols_ran; i_brow++) b_ran(i_brow,0)= b_samp(i_brow,0);

//b_ran = arma::sort(b_ran.t(),0);
//b_ran = b_ran.t();
//for(int i_brow=0; i_brow<ncols_ran; i_brow++){
//int k_temp = 0;
//for(int i_bcol=0; i_bcol<20; i_bcol++){
//if(b_ran(i_brow,i_bcol)<=b_last(i_brow,0)) k_temp = k_temp+1;
//}
//Rprintf("here");
//	if(k_temp == 0) k_temp=1;
//b_samp(i_brow,0)=b_ran(i_brow,20-k_temp);
//}
//return Rcpp::List::create(Rcpp::Named("b") = b_ran,Rcpp::Named("b2")=b_samp);



//Old gibbs sampler, w/o sorting; robust?
arma::mat ranvec0 = arma::randn(1,colsX);//Old
b_samp =  b_next +  XtX_inv_chol*ranvec0.t();//Old

//Update lambda
fits = X*b_samp;
ranscalar = rnorm(rowsX,0,1);
ranunif=  runif(rowsX,0,1);
for(int i_lambda=0; i_lambda<rowsX; i_lambda++) {
if(treat(i_lambda)==1) lambda(i_lambda) = 1;
if(treat(i_lambda)==0) {
//Generate inverse gaussian, as per Wikipedia
//R code: lambda[treat==0]<-1/rinvgauss(sum(treat==0),1/abs(1-(y[treat==0]*fits[treat==0])),1)
y_invg = ranscalar(i_lambda)*ranscalar(i_lambda);
mu_invg = (1-y(i_lambda)*fits(i_lambda,0));
if(mu_invg<0) mu_invg =  - mu_invg;
mu_invg=1/mu_invg;
lambda_invg = 1;
part1 = mu_invg+pow(mu_invg,2)*y_invg/2/lambda_invg;
part2 = 4*mu_invg*lambda_invg*y_invg+pow(mu_invg*y_invg,2);
part3 = mu_invg/lambda_invg/2*pow(part2,.5);
x_invg = part1 - part3;
if(ranunif(i_lambda)<= mu_invg/(mu_invg+x_invg)) {
lambda(i_lambda)=1/x_invg;//Reciprocal taken here
} else{
lambda(i_lambda)=x_invg/(mu_invg*mu_invg);//Reciprocal taken here
}
if(lambda(i_lambda)<0.0001) lambda(i_lambda) = 0.0001;
if(lambda(i_lambda)!=lambda(i_lambda)) lambda(i_lambda)=1/0.0001;
}

}//closes lambda loop

sumb=0;
for(int i_b=0; i_b<colsX; i_b++) sumb += pow(b_samp(i_b),2);
//R code:	nu<-1/rgamma(1,length(b)+.1,.1+sum(b^2))^.5

nu = rgamma(1, colsX+.1, .1+ sumb)(0);
nu = pow(nu, -.5);
if(param==1) nu = 100;



//Calculate effect
for(int i_marg=0; i_marg<rowsX; i_marg++){
margvec(i_marg, i_marg)=0;
if(treat(i_marg)==1) margvec(i_marg,i_marg) = 1;
if(1-y(i_marg)*fits(i_marg,0)> 0) margvec(i_marg, i_marg) = 1;

}

}

temp_coef= arma::pinv(arma::strans(X_uncond)*arma::diagmat(margvec)*X_uncond)*arma::strans(X_uncond)*arma::diagmat(margvec)*dv;

if(i_outer>=burnin){
effect(i_outer-burnin,0) = temp_coef(1,0);
for(int i_b=0; i_b<colsX; i_b++) b_run(i_outer-burnin,i_b) = b_samp(i_b,0);
for(int i_marg=0; i_marg<rowsX; i_marg++) margrun(i_outer-burnin,i_marg) = margvec(i_marg,i_marg);

}


}


//return Rcpp::List::create(Rcpp::Named("vec") = someVector,
//                          Rcpp::Named("lst") = someList,
//                          Rcpp::Named("vec2") = someOtherVector);


return Rcpp::List::create(Rcpp::Named("effect") = effect,
Rcpp::Named("b_post") = b_run,
Rcpp::Named("margin") = margrun//,
//Rcpp::Named("wtsmarg") = wtsrun,
//Rcpp::Named("pos") = posfit,
//Rcpp::Named("neg") = fits
);
}
