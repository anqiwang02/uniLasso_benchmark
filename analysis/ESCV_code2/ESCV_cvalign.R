#Compare CV errors between different types of alignments (table 1)

require("lars")
l1norm <- function(x) {
	return(sum(abs(x)))
	}

l2norm <- function(x) {
	return(sqrt(sum(x^2)))
	}
sigma=1
n=100
p=150
r=10
k=10
nruns=1000
errormean=matrix(0,nrow=4,ncol=3)
errorse=matrix(0,nrow=4,ncol=3)
for (iii in 1:4)  {
rho = c(0,0.2,0.5,0.9)[iii]
	
lars.list=list()
beta.list1=list()
beta.list2=list()
beta.list3=list()
	
Q = matrix(rho,nrow=p,ncol=p)		#covariance = 1 on diag and rho everywhere else
diag(Q) = 1
R=chol(Q)

error = matrix(0,nrow=nruns,ncol=3)
	
for (ii in 1:nruns) {

Z = matrix(rnorm(n*p),nrow=n)
X = Z %*% R

beta = c(runif(10,min=1/3,max=1),rep(0,p-r))


Y = c(X %*% beta + rnorm(n,sd=sigma))

####
#END GENERATE DATA
####

	#lambda, norm, fraction
	my.lars = lars(X,Y,normalize=FALSE,intercept=FALSE,use.Gram=FALSE)
	temp <- coef.lars(my.lars,s=0,mode="lambda")
	maxnorm = l1norm(temp)
	lambdavec = my.lars$lambda
	fracvec = seq(0,1,length.out=100)

	#Get pseudo solutions

	grps = cut(1:n,k,labels=FALSE)[sample(n)]
	for (i in 1:k) {
		omit = which(grps==i)
		lars.list[[i]] = lars(X[-omit,],Y[-omit],type="lasso",use.Gram=FALSE,normalize=FALSE,intercept=FALSE)
		temp <- coef.lars(lars.list[[i]],s=0,mode="lambda")
		maxnorm = min(maxnorm, l1norm(temp))
		}	

	#Get lambdavector and corresponding solutions
	normvec = seq(0,maxnorm*0.999,length=100)

	for (i in 1:k) {
		beta.list1[[i]] = t(coef.lars(lars.list[[i]],s=lambdavec, mode="lambda"))
		beta.list2[[i]] = t(coef.lars(lars.list[[i]],s=normvec, mode="norm"))
		beta.list3[[i]] = t(coef.lars(lars.list[[i]],s=fracvec, mode="fraction"))
		}	

	## CV

	run.l2pe.norm1 = rep(0,length(lambdavec))
	run.l2pe.norm2 = rep(0,100)
	run.l2pe.norm3 = rep(0,100)


	for (i in 1:k) {
		omit = which(grps==i)
		pe1 = Y[omit] - X[omit,] %*% beta.list1[[i]]
		pe2 = Y[omit] - X[omit,] %*% beta.list2[[i]]
		pe3 = Y[omit] - X[omit,] %*% beta.list3[[i]]
		l2pe1 = apply(pe1,2,l2norm)^2
		l2pe2 = apply(pe2,2,l2norm)^2
		l2pe3 = apply(pe3,2,l2norm)^2
		run.l2pe.norm1 = run.l2pe.norm1 + l2pe1
		run.l2pe.norm2 = run.l2pe.norm2 + l2pe2
		run.l2pe.norm3 = run.l2pe.norm3 + l2pe3
		}

	xval1 <- which.min(run.l2pe.norm1)
	xval2 <- which.min(run.l2pe.norm2)
	xval3 <- which.min(run.l2pe.norm3)

	betahat1 = c(coef.lars(my.lars,s=lambdavec[xval1],mode="lambda"))
	betahat2 = c(coef.lars(my.lars,s=normvec[xval2],mode="norm"))
	betahat3 = c(coef.lars(my.lars,s=fracvec[xval3],mode="fraction"))

	error[ii,1] = sqrt(sum((beta-betahat1)^2))
	error[ii,2] = sqrt(sum((beta-betahat2)^2))
	error[ii,3] = sqrt(sum((beta-betahat3)^2))

	
	}


errormean[iii,] = apply(error,2,mean)
errorse[iii,] = apply(error,2,sd)/sqrt(nruns)


}

errormean
errorse