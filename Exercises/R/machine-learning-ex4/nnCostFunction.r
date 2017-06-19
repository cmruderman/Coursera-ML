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

Theta1 <- weights$Theta1
Theta2 <- weights$Theta2

unrolled_theta1 <- as.matrix(as.vector(Theta1))
unrolled_theta2 <- as.matrix(as.vector(Theta2))

nn_params <- as.matrix(c(unrolled_theta1, unrolled_theta2))

sigmoidGradient <- function(z){	
	g <- matrix(0, dim(z)[1], dim(z)[2])
	return (sigmoid(z)*(1-sigmoid(z)))
}

nnCostFunction <- function(nn_params, input_layer_size,hidden_layer_size, num_labels, X, y, lambda){
	#reshape the Thetas

# 	% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# % for our 2 layer neural network

	Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))], nrow=hidden_layer_size, ncol=(input_layer_size + 1));
	Theta2 <- matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):dim(nn_params)[1]], nrow=num_labels, ncol=(hidden_layer_size + 1));

	m<-dim(X)[1]
	J <- 0;

	Theta1_grad <- matrix(0,dim(Theta1)[1], dim(Theta1)[2])
	Theta2_grad <- matrix(0,dim(Theta2)[1], dim(Theta2)[2])

# 	% Part 1: Feedforward the neural network and return the cost in the
# %         variable J. After implementing Part 1, you can verify that your
# %         cost function computation is correct by verifying the cost
# %         computed in ex4.m

	a1 <- cbind(matrix(1,m, 1), X); #include bias unit

	z2 <- a1%*%t(Theta1); #calculate z2
	a2 <- sigmoid(z2); #calculate g(z2)
	a2 <- cbind(matrix(1,m, 1), a2); #include bias unit a0(2)

	z3 <- a2%*%t(Theta2); #calculate z3
	a3 <- sigmoid(z3); #calculate g(z3) #h_theta is 5000x10
	h_theta <- a3

	y_vector <- matrix(0, m, num_labels)

	for(w in 1:m){
		y_vector[w, y[w]] <- 1; #$row vectors that represent the correct output,y_vector is 5000x10
	}

	left_term <- sum((-1*y_vector) * log(h_theta));
	right_term <- sum((1-y_vector) *log(1-h_theta));

	J <- (1/m)*(sum(left_term-right_term));

	Theta1_term <- sum(Theta1[1:dim(Theta1)[1],2:dim(Theta1)[2]]^2)
	Theta2_term <- sum(Theta2[1:dim(Theta2)[1],2:dim(Theta2)[2]]^2)

	regularization <- (sum(Theta1_term)+sum(Theta2_term))*(lambda/(2*m));
	J<-J+regularization;

# % Part 2: Implement the backpropagation algorithm to compute the gradients
# %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
# %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
# %         Theta2_grad, respectively. After implementing Part 2, you can check
# %         that your implementation is correct by running checkNNGradients
	
	tridelt_1<-0;
	tridelt_2<-0;

	delta_3<- a3-y_vector; #yk:{0,1} 

	z2 <- cbind(matrix(1, m, 1), z2) #include bias unit a0(2)
	delta_2 <- (delta_3%*%Theta2)*sigmoidGradient(z2)
	delta_2 <- delta_2[1:dim(delta_2)[1],2:dim(delta_2)[2]]

	tridelt_1 <- tridelt_1+(t(delta_2)%*%(a1)) # Same size as Theta1_grad (25x401)
	tridelt_2 <- tridelt_2+(t(delta_3)%*%(a2)) # Same size as Theta2_grad (10x26)


	Theta1_grad <- tridelt_1 / m; #for j=0, Dij=(1/m)*delta_ij
	Theta2_grad <- tridelt_2 / m; #for j=0, Dij=(1/m)*delta_ij

	Theta1_grad[1:dim(Theta1_grad)[1], 2:dim(Theta1_grad)[2]] <- Theta1_grad[1:dim(Theta1_grad)[1], 2:dim(Theta1_grad)[2]] + (lambda/m) * Theta1[1:dim(Theta1)[1], 2:dim(Theta1)[2]]
	Theta2_grad[1:dim(Theta2_grad)[1], 2:dim(Theta2_grad)[2]] <- Theta2_grad[1:dim(Theta2_grad)[1], 2:dim(Theta2_grad)[2]] + (lambda/m) * Theta2[1:dim(Theta2)[1], 2:dim(Theta2)[2]]

	unrolled_theta1_grad <- as.matrix(as.vector(Theta1_grad))
	unrolled_theta2_grad <- as.matrix(as.vector(Theta2_grad))

	grad <- as.matrix(c(unrolled_theta1_grad, unrolled_theta2_grad))

	newList <- list("cost" = J, "theta" = grad);
	return (newList);
}

returnedList <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0);

J<-returnedList$cost
grad<-returnedList$theta

cat("Cost at parameters (loaded from ex4weights): (this value should be about 0.287629):", J);

returnedList <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1);

J<-returnedList$cost
grad<-returnedList$theta


cat("\nCost at parameters (loaded from ex4weights): (this value should be about 0.383770) ", J);













