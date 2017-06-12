df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex1/ex1/ex1data1.txt", header = FALSE, sep = ",")
colnames(df)[1] <- "Population"
colnames(df)[2] <- "Profit"
#matrix(0, n, m) #creates a matrix with n lines and m columns, filled with zeros.

X <- cbind(a=1, df$Population);
y<-df[,c("Profit")];
m<-length(df$Profit);
theta<-matrix(0,2,1);

computeCost <- function(matrixA, matrixB, theta_matrix, m){
	predictions<-matrixA%*%theta_matrix;
	squaredErrors<-(predictions-matrixB)^2;
	J=(1/(2*m))*sum(squaredErrors);
	return (J);
}

new_theta <- matrix(c(-1, 2), ncol = 1);

cat("With theta=[0;0] cost computed= ", computeCost(X, y, theta, m));
print("Expected cost value (approx) 32.07");

cat("With theta=[0;0] cost computed= ", computeCost(X, y, new_theta, m));
print("Expected cost value (approx) 54.24");

