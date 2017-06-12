library(R.matlab)
library(sigmoid)

pathname_data<-"/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex3/ex3/ex3data1.mat"
data <- readMat(pathname_data)

X <- data$X
y <- data$y

m <- dim(X)[1]

pathname_weights<-"/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex3/ex3/ex3weights.mat"
weights <- readMat(pathname_weights)

Theta1 <- weights$Theta1
Theta2 <-weights$Theta2


predict <-function(Theta1, Theta2, X){
	m<-dim(as.matrix(X))[1]
	p <- matrix(0, m, 1)
	a1<- cbind(rep(1,m), X) #include bias unit
	z2<-a1 %*% t(Theta1) #calculate z2
	a2<-sigmoid(z2) #calculate g(z2)
	hidden_layer <- cbind(rep(1,m), a2) #include bias unit a0(2)
	z3 <- hidden_layer %*% t(Theta2) #calculate z3
	a3 <- sigmoid(z3); #calculate g(z3)
	p <- apply(a3, 1, which.max)
	return (p)
}

p <- predict(Theta1, Theta2, X)
cat("Train Accuracy: ", mean((p == y))*100);