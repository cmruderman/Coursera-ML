df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex1/ex1/ex1data1.txt", header = FALSE, sep = ",")
colnames(df)[1] <- "Population"
colnames(df)[2] <- "Profit"
plot(df$Population, df$Profit, type="p", main="Pop vs $", ylab="Profit", xlab="Population")