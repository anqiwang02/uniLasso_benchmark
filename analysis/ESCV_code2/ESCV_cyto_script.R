qX = read.delim("qX.txt",header=FALSE)
qY = read.delim("qY.txt",header=FALSE)

find.mode = function(x) {
	as.numeric(names(sort(-table(x)))[1])}


Xnames = c('cAMP','AKT','ERK1','ERK2','Ezr/Rdx','GSK3A','GSK3B','JNK lg','JNK sh','MSN','p38','p40Phox','NFkB p65','PKCd','PKCmu2','RSK','Rps6','SMAD2','STAT1a','STAT1b','STAT3','STAT5')
Ynames =c('G-CSF','IL-1a','IL-6','IL-10','MIP-1a','RANTES','TNFa')

#create no-nans X,Y lists
nnX=list()
nnY=list()

for (i in 1:7) {
	temp = cbind(qY[,i],qX)
	nanindex = which(is.nan(apply(temp,1,sum)))
	if(length(nanindex>0)) {
  	  nnX[[i]] = qX[-nanindex,]
	  nnY[[i]] = qY[-nanindex,i]	
      } else {
      	nnX[[i]] = qX
      	nnY[[i]] = qY[,i]
      	}	
	}

####

cyt.index=1 #1 through 7

X = as.matrix(nnX[[cyt.index]])
Y = nnY[[cyt.index]]

X = scale(X)
Y = c(scale(Y))


##### ESCV
source("ESCV_glmnet_vf.R")

my.escv.glmnet <- escv.glmnet(X,Y)
orig_betahat = as.matrix(my.escv.glmnet[['glmnet']][['beta']])
betahats = orig_betahat[,my.escv.glmnet$selindex]



