install.packages("sigmoid")
library(sigmoid)

df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex2/ex2/ex2data1.txt", header = FALSE, sep = ",")
colnames(df)[1] <- "Exam 1"
colnames(df)[2] <- "Exam 2"
colnames(df)[3] <- "Result"
X <- as.matrix(df[,c("Exam 1","Exam 2")])
y <- df[,c("Result")]
m<-dim(X)[1]
n<-dim(X)[2]
initial_theta <- matrix(0,n+1,1)

costFunction <- function(theta, matrix_X, matrix_Y){
	n<-dim(X)[2]
	matrix_X <- cbind(a=1, matrix_X)
	grad <-matrix(0,n+1,1)
	subtraction <- matrix(0,n+1,1)
	argument <- t(t(theta)%*%t(matrix_X))
	g <- sigmoid(argument);
	sumation <- sum((t(-1*matrix_Y)%*%log(g))-(t(1-matrix_Y)%*%log(1-g)));
	J=(1/m)*(sumation);
	for(iter in 1:m){
		for (feature in 1:length(theta)){
			subtraction[feature]=subtraction[feature]+(t(g[iter]-matrix_Y[iter])*matrix_X[iter,feature]);
		}
	}
	grad=(1/m)*subtraction;
	newList <- list("cost" = J, "theta" = grad);
	return (newList);
}

returnedList <- costFunction(initial_theta, X, y);

cat("Cost at initial theta (zeros): ", returnedList$cost);
print("Expected cost (approx): 0.693");
cat("Gradient at initial theta (zeros):", returnedList$theta);
print("Expected gradients (approx): -0.1000 -12.0092 -11.2628");

res <- optim(par = initial_theta, 
             fn = function(t) costFunction(t, X, y)$cost,
             gr = function(t) costFunction(t, X, y)$theta,
             method = "BFGS", control = list(maxit = 400))

cost <- res$value;
theta <- res$par

cat("Cost at theta found by fminunc: ", cost);
print("Expected cost (approx): 0.203");
cat("theta: ", theta);
print("Expected theta (approx): -25.161 0.206 0.201");

#45 on first exam, 85 on second

predict <- function(theta, matrix_X){
	m<-dim(matrix_X)[1];
	matrix_X <- cbind(a=1, matrix_X)
	argument<-t(t(theta)%*%t(matrix_X));
	G<-sigmoid(argument);
	p<-(G>=.5);
	return (p);
}

prob <- sigmoid(matrix(c(1,45,85), nrow=1, ncol=3)%*%theta)
p <- predict(theta, X)

cat("Train Accuracy: ", mean((p == y)) * 100);
print("Expected accuracy (approx): 89.0")



