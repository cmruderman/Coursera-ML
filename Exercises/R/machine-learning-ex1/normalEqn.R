install.packages("corpcor")
library(corpcor)
print("Solving with normal equations");

df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex1/ex1/ex1data2.txt", header = FALSE, sep = ",");
colnames(df)[1] <- "Feet_Squared";
colnames(df)[2] <- "Bedrooms";
colnames(df)[3] <- "Price";
#matrix(0, n, m) #creates a matrix with n lines and m columns, filled with zeros.

X <- as.matrix(df[,c("Feet_Squared","Bedrooms")]);
X <- cbind(a=1, X);
y <- as.matrix(df[,c("Price")]);
m <- length(df$Price);


theta <-pseudoinverse(t(X)%*%X)%*%t(X)%*%y;
price <- matrix(c(1, 1650, 3), 1,3)%*%theta; #1x3 matrix to hold data




cat("Predicted price of a 1650 sq-ft, 3 br house ", price);
