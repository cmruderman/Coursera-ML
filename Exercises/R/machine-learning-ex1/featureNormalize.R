df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex1/ex1/ex1data2.txt", header = FALSE, sep = ",");
colnames(df)[1] <- "Feet_Squared";
colnames(df)[2] <- "Bedrooms";
colnames(df)[3] <- "Price";
#matrix(0, n, m) #creates a matrix with n lines and m columns, filled with zeros.

X <- df[,c("Feet_Squared","Bedrooms")];

X <- scale(X); #only thing you need for feature normalization
