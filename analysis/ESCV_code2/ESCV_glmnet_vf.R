
####
#	my.escv.glmnet <-escv.glmnet(X,Y)
#	orig_betahat = as.matrix(my.escv.glmnet[['glmnet']][['beta']])

# Note: The orig_betahat here is from the model fit on the full data, not the k=10 folds.
# The k=10 folds each have their own beta estimates (see beta.list in escv.glmnet), but only the full-data solution path is returned in my.escv.glmnet$glmnet$beta.

#betahat = orig_betahat[,my.escv.glmnet$selindex]



####


escv.glmnet <- function(X,Y,k=10, nlambda=100) {

	require(glmnet)	
	n = dim(X)[1]
	p = dim(X)[2]
	if (length(Y)!=n) {warning("length(Y) != nrow(X)")}
	
	glmnet.list = list()
	beta.list = list()
	Xb.list = list()

	#Get overall solution path (solve for beta at all lambda values)

	orig_glmnet = glmnet(X,Y, family="gaussian", alpha=1, nlambda=nlambda, standardize=FALSE, intercept=FALSE)

	lambdavec = orig_glmnet$lambda #all the lambda values 
	orig_betahat = as.matrix(orig_glmnet$beta) #matrix of coefficients beta, each column corresponds to a lambda value

	#Get pseudo solutions

	grps = cut(1:n,k,labels=FALSE)[sample(n)] #divides the sequence 1:n into k groups and assigns each element a group number (1 to k). By setting labels=FALSE, you get numeric group assignments instead of text labels
	
	for (i in 1:k) {
		omit = which(grps==i) # For each fold i, omit contains the indices for the current validation set.
		#Fit Lasso Model on Training Data (alpha=1 means lasso regression)
		#glmnet.list stores the fitted glmnet model 
		glmnet.list[[i]] = glmnet(X[-omit,,drop=FALSE],Y[-omit], family="gaussian", alpha=1, lambda=lambdavec, standardize=FALSE, intercept=FALSE)
		# Extract Coefficients for Each Lambda
		beta.list[[i]] = as.matrix(glmnet.list[[i]]$beta)
		# Compute Predicted Values at varying lambda. contains predictions for all data points, not just the validation fold. You may want to subset these if you only need predictions for the omitted fold.
		Xb.list[[i]] = (X %*% beta.list[[i]])
		}	

	## CV

	# This matrix is used to store results (such as prediction errors) for each combination of lambda value and cross-validation fold.
	l2pe = matrix(0,length(lambdavec),k)

	for (i in 1:k) {
		omit = which(grps==i)
		pe = Y[omit] - X[omit,,drop=FALSE] %*% beta.list[[i]] # X*beta is a matrix where each column represents predictions for a different lambda value
		# pe is the prediction error matrix for the omitted fold i, where each column corresponds to a different lambda value, and each row corresponds to an observation in the omitted fold 
		l2pe[,i] = apply(pe^2,2,mean) # calculate the mean squared prediction for each lambda value (column in pe) across all observations in the omitted fold (rows)
		}

	CV = apply(l2pe,1,mean) # average the mean squared prediction errors across all k folds (across each row) for each lambda value
	CV.index = which.min(CV) # index of the lambda value that gives the minimum average prediction error across all folds

	#Compute ES metric

	beta.sum = beta.list[[1]]
	# coefficients sum across folds for each lambda value
	for (i in 2:k) {
		beta.sum = beta.sum + beta.list[[i]]
		}
	beta.mean <- beta.sum /k
	Xb.mean <- (X %*% beta.mean) # predicted values on the full dataset using the average coefficients

	l2diffX.matrix = matrix(0,nrow=k,ncol=length(lambdavec)) # folds on rows and lambda values on columns
	#See the ES formula in the paper 
	for (i in 1:k) {
		l2diffX.matrix[i,]=apply((Xb.list[[i]] - Xb.mean)^2,2,sum)
		}
	# denominator in the ES formula 
	l2diffX <- apply(l2diffX.matrix,2,mean)

	ES = l2diffX/apply(Xb.mean^2,2,sum)

	# check for convergence
	ESgrad = ES[2:length(ES)]-ES[1:(length(ES)-1)]
	con.index = which(ESgrad < 0)
	if(length(con.index)==0) {
		con.index = 1
		} else {con.index = con.index[1]}	

	# Find min after convergence before CV choice
	if(con.index > CV.index) {
		con.index = CV.index
		}
		
	ESCV.index = which.min(ES[con.index:CV.index]) + con.index - 1
	# +con.index -1 to adjust for the offset in indexing due to subsetting ES from con.index to CV.index


	# BIC, extended BIC
	l0norm <- function(x){return(sum(x!=0))}
	df = apply(orig_betahat,2,l0norm)
	RSS = apply( (Y - X %*% orig_betahat)^2 , 2 ,sum )

	BIC = n*log(RSS) + log(n)*df
	EBIC1 = n*log(RSS) + log(n)*df + 2 * 0.5 * log(choose(p,df)) #Gamma = 0.5 for high dimensional case
	EBIC2 = n*log(RSS) + log(n)*df + 2 * 1 * log(choose(p,df)) #Gamma = 1 for ultra high dimensional case

	BIC.index = which.min(BIC)
	EBIC.index = c(which.min(EBIC1),which.min(EBIC2))	

	results = list()
	results$glmnet <- orig_glmnet
	results$selindex <- c(ESCV.index,CV.index,BIC.index, EBIC.index)
	names(results$selindex) <- c("ESCV","CV","BIC","EBIC1","EBIC2")		
	return(results)
	
	}


#### repeated CV for v=2 repeated 5 times

rcv.glmnet <- function(X,Y, k=2, r=5, nlambda=100) {
	
	require(glmnet)	
	n = dim(X)[1]
	p = dim(X)[2]
	if (length(Y)!=n) {warning("length(Y) != nrow(X)")}
	
	glmnet.list = list()
	beta.list = list()
	Xb.list = list()

	#Get overall solution path

	orig_glmnet = glmnet(X,Y, family="gaussian", alpha=1, nlambda=nlambda, standardize=FALSE, intercept=FALSE)

	lambdavec = orig_glmnet$lambda 
	orig_betahat = as.matrix(orig_glmnet$beta)

	#Get pseudo solutions
	l2per = matrix(0,length(lambdavec),k*r)

	for (ii in 1:r) {

	grps = cut(1:n,k,labels=FALSE)[sample(n)]
	
	for (i in 1:k) {
		omit = which(grps==i)
		glmnet.list[[i]] = glmnet(X[-omit,,drop=FALSE],Y[-omit], family="gaussian", alpha=1, lambda=lambdavec, standardize=FALSE, intercept=FALSE)
		beta.list[[i]] = as.matrix(glmnet.list[[i]]$beta)
		Xb.list[[i]] = (X %*% beta.list[[i]])
		}	

	## CV

	l2pe = matrix(0,length(lambdavec),k)

	for (i in 1:k) {
		omit = which(grps==i)
		pe = Y[omit] - X[omit,,drop=FALSE] %*% beta.list[[i]]
		l2pe[,i] = apply(pe^2,2,mean)
		}

	l2per[,((ii-1)*k+1):(ii*k)]=l2pe

	}


	rCV = apply(l2per,1,mean)
	rCV.index = which.min(rCV)

	results = list()
	results$glmnet <- orig_glmnet
	results$selindex <- rCV.index
	return(results)	
}


	