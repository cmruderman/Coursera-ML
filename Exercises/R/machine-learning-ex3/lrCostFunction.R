install.packages("R.matlab")
install.packages("sigmoid")
library(R.matlab)
library(sigmoid)

pathname<-"/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex3/ex3/ex3data1.mat"
data <- readMat(pathname)

theta_t <- matrix(c(-2, -1, 1, 2), ncol = 1)
X_t <-matrix(c(1, 1, 1, 1, 1, .1, .2, .3, .4, .5, .6 ,.7,.8,.9,1, 1.1, 1.2,1.3,1.4,1.5), ncol=4);
y_t <- matrix(c(1,0,1,0,1), ncol = 1)
lambda_t=3


lrCostFunction <- function(matrixA, matrixB, theta_matrix, lambda){
	m<-length(matrixB);
	length_of_theta <- length(theta_matrix)[1];
	argument<-t(t(theta_matrix)%*%t(matrixA));  
	g<-sigmoid(argument);

	#regularized cost
	sumation<-sum((t(-1*matrixB)%*%log(g))-(t(1-matrixB)%*%log(1-g)));
	J<-((1/m)*(sumation))+(lambda/(2*m))%*%sum((theta_matrix[2:length_of_theta,]^2));

	#regularized gradient
	temp<-theta_matrix;
	temp[1]<-0;
	grad<-(1/m)*(t(matrixA)%*%(g-matrixB))+(lambda/m)*temp;
	newList <- list("cost" = J, "theta" = grad);
	return (newList);
}


returnedList <- lrCostFunction(X_t, y_t, theta_t, lambda_t);

J<-returnedList$cost
grad<-returnedList$theta

cat("Cost: ", J);
print("Expected cost: 2.534819");
cat("Gradients: ", grad);
print("Expected gradients: 0.146561 -0.548558 0.724722 1.398003");
