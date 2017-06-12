X <- matrix(c(1, 2104, 5, 1, 45, 1,1416,3,2,40,1,1534,3,2,30,1,852,2,1,36), nrow=4, ncol=5, byrow=TRUE)

Y <- c(460, 232, 315, 178)

A <- solve(X %*% t(X)) #gets the inverse of X*X^T
result <- t(X) %*% solve(X %*% t(X)) %*% Y # R doesnt like the order so change so dimensions work

dimnames(result) = list(c("row1", "row2", "row3","row4", "row5"), c("col1")) 

	#this is how output will be formatted
    #    col1
# row1 188.4003191
# row2   0.3866255
# row3 -56.1382494
# row4 -92.9672535
# row5  -3.7378190
