library(R.matlab)
library(sigmoid)

pathname<-"/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex3/ex3/ex3data1.mat"
data <- readMat(pathname)

X <- data$X
y <- data$y

num_labels <- 10      
lambda <- .1

lrCostFunction <- function(theta_matrix, matrixA, matrixB, lambda){
	theta_matrix <- as.matrix(theta_matrix)
	m<-length(matrixB);
	argument<-t(t(theta_matrix)%*%t(matrixA));  
	g<-sigmoid(argument);

	#regularized cost
	sumation<-sum((t(-1*matrixB)%*%log(g))-(t(1-matrixB)%*%log(1-g)));
	J<-((1/m)*(sumation))+(lambda/(2*m))%*%sum((as.matrix(theta_matrix[2:dim(theta_matrix)[1],])^2));

	#regularized gradient
	temp<-theta_matrix;
	temp[1]<-0;
	grad<-(1/m)*(t(matrixA)%*%(g-matrixB))+(lambda/m)*temp;
	newList <- list("cost" = J, "theta" = grad);
	return (newList);
}


oneVsAll <- function(X, y, num_labels, lambda){
	m<-dim(X)[1]
	n<-dim(X)[2]
	all_theta <- matrix(0,num_labels,n+1)
	X <- cbind(rep(1,m), X)
	initial_theta <- matrix(0, n+1, 1)
	 for (c in 1:num_labels){
	 	res <- optim(par = initial_theta, 
             fn = function(t) lrCostFunction(t, X, y==c, lambda)$cost,
             gr = function(t) lrCostFunction(t, X, y==c, lambda)$theta,
             method = "BFGS", control = list(maxit = 50))
	    theta <- res$par
	    all_theta[c,] <- t(theta)
	  }
  	return (all_theta)
}

returnedList <- oneVsAll(X, y, num_labels, lambda)

predict <- function(theta, X){
	m <- dim(X)[1]
	num_labels <- dim(theta)[1]
	p <- as.matrix(rep(0,dim(X)[1]))
	X <- cbind(matrix(1,m,1), X)

	a <- sigmoid(X %*% t(theta))
	p <- apply(a, 1, which.max)
	return (p);
}

p <- predict(returnedList, X)

cat("Train Accuracy: ", mean((p == y))*100);


