library(R.matlab)
library(sigmoid)

pathname_data<-"/Users/cmr/Desktop/Machine Learning/Class/R/ex4/ex4data1.mat"
data <- readMat(pathname_data)

pathname_weights<-"/Users/cmr/Desktop/Machine Learning/Class/R/ex4/ex4weights.mat"
weights <- readMat(pathname_weights)

input_layer_size  <- 400;  # 20x20 Input Images of Digits
hidden_layer_size <- 25;   # 25 hidden units
num_labels <- 10;          # 10 labels, from 1 to 10  

X <- data$X
y <- data$y

randInitializeWeights <- function(L_in, L_out){	
	W<-matrix(0, L_out, 1+L_in)
	epsilon<-.12
	return (matrix(runif(L_out*(L_in+1)), L_out, L_in+1)); #returns matrix that is L_out x (L_in+1)
}

initial_Theta1 <- randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 <- randInitializeWeights(hidden_layer_size, num_labels)


unrolled_theta1 <- as.matrix(as.vector(Theta1))
unrolled_theta2 <- as.matrix(as.vector(Theta2))

initial_nn_params <- as.matrix(c(unrolled_theta1, unrolled_theta2))

print("\nChecking Backpropagation (w/ Regularization) ... \n")

# Also output the costFunction debugging values
returnedList  <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 3);

J<-returnedList$cost
grad<-returnedList$theta

cat("Cost at (fixed) debugging parameters (for lambda = 3, this value should be about 0.576051)", J);







