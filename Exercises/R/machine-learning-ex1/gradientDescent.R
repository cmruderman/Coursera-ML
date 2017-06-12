source("computeCost.R");
source("gradientDescent.R");

print("Running Gradient Descent...");
df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex1/ex1/ex1data1.txt", header = FALSE, sep = ",")
colnames(df)[1] <- "Population"
colnames(df)[2] <- "Profit"
#matrix(0, n, m) #creates a matrix with n lines and m columns, filled with zeros.

X <- cbind(a=1, df$Population);
y<-df[,c("Profit")];
m<-length(df$Profit);
theta<-matrix(0,2,1);
gradientDescent <- function(matrixA, matrixB, theta_matrix, alpha, iterations, m){
	for(i in 1:iterations){
	    hypothesis <- matrixA %*% theta_matrix;  
	    error <- hypothesis - matrixB;   #X*theta-y
	    delta <- (1 / m) * (t(matrixA) %*% error);
	    theta_matrix <- theta_matrix - alpha*delta;
	    J_history[i] <- computeCost(matrixA, matrixB, theta_matrix);

	}
	return (theta_matrix);
}

iterations <- 1500;
alpha <- 0.01;

J_history <- matrix( rep(0, len=iterations), nrow = iterations);

theta <- gradientDescent(X, y, theta, alpha, iterations, m);

cat("Theta found by gradient descent", theta);
print("Expected theta values (approx)\n");
print("-3.6303 1.1664");


print("Running Gradient Descent...");
df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex1/ex1/ex1data1.txt", header = FALSE, sep = ",")
colnames(df)[1] <- "Population"
colnames(df)[2] <- "Profit"
#matrix(0, n, m) #creates a matrix with n lines and m columns, filled with zeros.

X <- cbind(a=1, df$Population);
y<-df[,c("Profit")];
m<-length(df$Profit);
theta<-matrix(0,2,1);
gradientDescent <- function(matrixA, matrixB, theta_matrix, alpha, iterations, m){
	for(i in 1:iterations){
	    hypothesis <- matrixA %*% theta_matrix;  
	    error <- hypothesis - matrixB;   #X*theta-y
	    delta <- (1 / m) * (t(matrixA) %*% error);
	    theta_matrix <- theta_matrix - alpha*delta;
	    J_history[i] <- computeCost(matrixA, matrixB, theta_matrix);

	}
	return (theta_matrix);
}

iterations <- 1500;
alpha <- 0.01;

J_history <- matrix( rep(0, len=iterations), nrow = iterations);

theta <- gradientDescent(X, y, theta, alpha, iterations, m);

cat("Theta found by gradient descent", theta);
print("Expected theta values (approx)\n");
print("-3.6303 1.1664");

# plot(df$Population, df$Profit, type="p", main="Pop vs $", ylab="Profit", xlab="Population")
# par(new=TRUE); #equivalent to hold on 
# plot(X[,2], X%*%theta, type="l", ylab="", xlab="");

#multi gradient descent

df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex1/ex1/ex1data2.txt", header = FALSE, sep = ",");
colnames(df)[1] <- "Feet_Squared";
colnames(df)[2] <- "Bedrooms";
colnames(df)[3] <- "Price";
#matrix(0, n, m) #creates a matrix with n lines and m columns, filled with zeros.

X <- df[,c("Feet_Squared","Bedrooms")];
X <- scale(X); #only thing you need for feature normalization
X <- cbind(a=1, X);
y <-df[,c("Price")];
m <- length(df$Price);

alpha <- 0.1;
num_iters <- 100;
theta<-matrix(0,3,1);

theta <- gradientDescent(X, y, theta, alpha, num_iters, m);
price <- matrix(c(1, 1650, 3), ncol=3)*theta; 

cat("Predicted price of a 1650 sq-ft, 3 br house: $", price[2]);




