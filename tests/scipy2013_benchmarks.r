require(ggplot2)
require(reshape2)
require(foreach)

pdfPlot <- function(p, filename, width=6, height=4, units="in") {
  ggsave(filename, p, width=width, height=height, units=units)
}

# Dense matrix comparison. 
numElements <- c(1e2, 1e3, 1e4, 1e5, 2.5e6, 1e6, 5e6, 7.5e6, 1e7, 1e8)
rowSizes <- round(sqrt(numElements))
colSizes <- rowSizes

res <- vector(mode="character", length=length(rowSizes))
for (i in 1:length(rowSizes)) {
  res[i] <- system(paste("scipy_bench -m", rowSizes[i], "-n", colSizes[i], 
    "--nu 10 --csv"), intern=TRUE)
}

denseTiming <- as.data.frame(cbind(numElements, 
  matrix(as.numeric(unlist(strsplit(res, ","))), ncol=3, byrow=TRUE)))

names(denseTiming) <- c("Elements", "IRLB", "SVD", "Abs.Error")
denseTiming <- denseTiming[,-4]

denseTiming <- melt(denseTiming, id=c("Elements"))
names(denseTiming) <- c("Elements", "Algorithm", "Seconds")
p <- qplot(Elements, Seconds, data=denseTiming, color=Algorithm, geom="line",
  size=I(1))
pdfPlot(p, "DenseMatrixComparison.pdf")

# Dense matrix scaling in the number of singular vectors.
numElements <- c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
rowSizes <- round(sqrt(numElements))
colSizes <- rowSizes
numVec <- c(1, 5, 10, 15, 20)
nuTiming <- foreach (i=1:length(rowSizes), .combine=rbind) %do% {
  foreach (nu=numVec, .combine=c) %do% {
    as.numeric(system(paste("scipy_bench -m", rowSizes[i], "-n", 
      colSizes[i], "--nu", nu, "--csv --irlb-only"), intern=TRUE))
  }
}
nuTiming <- as.data.frame( cbind(numElements, nuTiming) )
names(nuTiming) <- c("Elements", as.character(numVec))
nuTiming <- melt(nuTiming, id="Elements")
names(nuTiming) <- c("Elements", "nu", "Seconds")
p <- qplot(Elements, Seconds, data=nuTiming, color=nu, geom="line", size=I(1))
pdfPlot(p, "DenseMatrixNuScaling.pdf")

# Sparse matrix scaling in the number of singular values with 95% 
# sparsity.
numElements <- c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
rowSizes <- round(sqrt(numElements))
colSizes <- rowSizes
numVec <- c(1, 5, 10, 15, 20)
nuTiming <- foreach (i=1:length(rowSizes), .combine=rbind) %do% {
  foreach (nu=numVec, .combine=c) %do% {
    as.numeric(system(paste("scipy_bench -m", rowSizes[i], "-n", 
      colSizes[i], "--nu", nu, "--csv --irlb-only -s 0.99"), intern=TRUE))
  }
}
nuTiming <- as.data.frame( cbind(numElements, nuTiming) )
names(nuTiming) <- c("Elements", as.character(numVec))
nuTiming <- melt(nuTiming, id="Elements")
names(nuTiming) <- c("Elements", "nu", "Seconds")
p <- qplot(Elements, Seconds, data=nuTiming, color=nu, geom="line", size=I(1))
pdfPlot(p, "SparseMatrixNuScaling.pdf")


