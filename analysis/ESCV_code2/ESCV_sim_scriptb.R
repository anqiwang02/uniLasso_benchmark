BegTime=Sys.time()

source("ESCV_glmnet_vf.R")

l2norm <- function(x){return(sqrt(sum(x^2)))}

nruns = 1000

score.b <- matrix(0,nrow=nruns,ncol=5)
score.pred <- matrix(0,nrow=nruns,ncol=5)
true.var.picked <- matrix(0,nrow=nruns,ncol=5)
noise.var.picked <- matrix(0,nrow=nruns,ncol=5)

colnames(score.b)=c("ESCV","CV","BIC","EBIC1","EBIC2")
colnames(score.pred)=c("ESCV","CV","BIC","EBIC1","EBIC2")

####
#GENERATE DATA
####

counter=0
for (rho in c(0.3, 0.5,0.9)) {
for (sigma in c(0.5,1,2,4)) {
for (p in c(50,150,300,500)) {

n=100
r=10
Q = matrix(0,nrow=p,ncol=p)	

### for block design

for (i in 1:p) {
	for (j in 1:p) {
		if ((i %% 10)==(j %% 10)) {Q[i,j]=rho}
		}
	}

#### for toeplitz design
#
# for (i in 1:p) {
# 	for (j in 1:p) {
#		Q[i,j] = rho^abs(i-j)
#		}
#	}


diag(Q) = 1
R=chol(Q)


for (ii in 1:nruns) {

Z = matrix(rnorm(n*p),nrow=n)
X = Z %*% R

#randomize columns (preserves cov struc but randomizes which ones are real)
temp <- sample(p,p)

beta=rep(0,p)
beta[temp] = c(runif(10,min=1/3,max=1),rep(0,p-r))


Y = c(X %*% beta + rnorm(n,sd=sigma))

####
#END GENERATE DATA
####


	my.escv.glmnet <-escv.glmnet(X,Y)
	orig_betahat = as.matrix(my.escv.glmnet[['glmnet']][['beta']])
	betahats = orig_betahat[,my.escv.glmnet$selindex]


score.b[ii,]<-apply(beta-betahats,2,l2norm)
score.pred[ii,]<- sqrt(diag(t(beta - betahats ) %*% Q %*% (beta - betahats)))
true.var.picked[ii,] <- apply(betahats[temp,][1:r,] != 0,2,sum)
noise.var.picked[ii,] <- apply(betahats[temp,][(r+1):p,] != 0 ,2,sum)
}

counter = counter + 1

fname = paste("escv_outb_",counter,sep="")

save(n,p,rho,sigma,score.b,score.pred,true.var.picked,noise.var.picked,file=fname)


}
}
}

EndTime=Sys.time()
RunTime=EndTime-BegTime
print(RunTime)

