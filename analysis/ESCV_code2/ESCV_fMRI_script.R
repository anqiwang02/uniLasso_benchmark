load("ESCV_fMRI.RData")
source("ESCV_glmnet_vf.R")

score <- matrix(0,nrow=500,ncol=7)
p.picked <- matrix(0,nrow=500,ncol=7)

beta.ESCV <- matrix(0,nrow=10409,ncol=500)
beta.CV <- matrix(0,nrow=10409,ncol=500)
beta.BIC <- matrix(0,nrow=10409,ncol=500)
beta.rCV <- matrix(0,nrow=10409,ncol=500)
beta.rCV2 <- matrix(0,nrow=10409,ncol=500)

X = train.X

for(i in 1:500){

	Y = train.Y[,i]

	my.escv.glmnet <-escv.glmnet(X,Y)
	orig_betahat = as.matrix(my.escv.glmnet[['glmnet']][['beta']])
	betahats = orig_betahat[,my.escv.glmnet$selindex]

	my.rcv.glmnet <- rcv.glmnet(X,Y,k=2,r=5)
	orig_betahat = as.matrix(my.rcv.glmnet[['glmnet']][['beta']])
	betahats = cbind(betahats, orig_betahat[,my.rcv.glmnet$selindex])

	my.rcv.glmnet <- rcv.glmnet(X,Y,k=5,r=2)
	orig_betahat = as.matrix(my.rcv.glmnet[['glmnet']][['beta']])
	betahats = cbind(betahats, orig_betahat[,my.rcv.glmnet$selindex])

	Yhat = val.X %*% betahats
	score[i,] <- cor(Yhat,val.Y[,i])
	p.picked[i,] <-  apply(betahats != 0,2,sum) 

	beta.ESCV[,i] = betahats[,1]
	beta.CV[,i] = betahats[,2]
	beta.BIC[,i]= betahats[,4]
	beta.rCV[,i]= betahats[,6]
	beta.rCV2[,i] = betahats[,7]

	}


