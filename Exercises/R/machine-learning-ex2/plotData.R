df <- read.table("/Users/cmr/Desktop/Machine Learning/Class/Octave/machine-learning-ex2/ex2/ex2data1.txt", header = FALSE, sep = ",")
colnames(df)[1] <- "Exam 1"
colnames(df)[2] <- "Exam 2"
colnames(df)[3] <- "Result"
X <- as.matrix(df[,c("Exam 1","Exam 2")]);
y <- df[,c("Result")]

pos<-which(y==1);
neg<-which(y==0);

plot(X[pos, 1], X[pos, 2], col="green", pch=4,type="p", xlim=range(20:100), ylim=range(20:100), main="Exam 1 vs Exam 2", ylab="Exam 2", xlab="Exam 1")
legend("bottomleft", legend=c("Admitted", "Not Admitted"),
       col=c("green", "red"), lty=1:2, cex=0.8)
par(new=TRUE)
points(X[neg, 1], X[neg, 2], col="red")