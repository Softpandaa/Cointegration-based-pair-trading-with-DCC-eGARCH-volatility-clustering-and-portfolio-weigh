# Pairs information ------
pair_description = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\pair_df.csv", row.names=1)
pair_description$pair_name = paste(pair_description$Ticker.1, pair_description$Ticker.2)
pairs.name = pair_description$pair_name
pairs.no = length(pair_description$pair_name); pairs.no

# Training spreads pre-processing ------
## First date: 1/30/2012, Last date: 8/5/2021
library(xts)
train.pair_spread = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\train_spread.csv", row.names=1)
train.pair_spread = as.xts(train.pair_spread) # Convert spreads to time series
train.half_life = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\train_half_life.csv", row.names=1)
for (col in 1:pairs.no){
  window = train.half_life$Half.Life[col]
  train.pair_spread[,col] = rollmean(train.pair_spread[,col], k = window, align = "right", fill = NA)
} # Convert to rolling spreads based on half_life
train.spread_scaled = scale(na.omit(train.pair_spread), center = TRUE, scale = TRUE) # Standardization for DCC_GARCH training
train.length = nrow(train.spread_scaled); train.length

list = rep(NA, 25)
for (p in 1:5) {
  for (q in 1:5) {
    list[q+5*(p-1)] = paste0("(",p,",",q,")")
  }
}
aic = matrix(NA, nrow=pairs.no, ncol=25, dimnames = list(pairs.name, list))
bic = matrix(NA, nrow=pairs.no, ncol=25, dimnames = list(pairs.name, list))
library(rugarch)
for (pair in 1:pairs.no){
  spread = train.spread_scaled[,pair]
  for (p in 1:5) {
    for (q in 1:5) {
      model <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(p, q)),
                          mean.model = list(armaOrder = c(1, 1), include.mean = FALSE),distribution.model = "std")
      fitted_model <- ugarchfit(model, data = spread)
      aic[pair,q+5*(p-1)] = infocriteria(fitted_model)[1]
      bic[pair,q+5*(p-1)] = infocriteria(fitted_model)[2]
    }
  }
}
AIC_table = na.omit(aic[,-20])
BIC_table = na.omit(bic[,-20])
aic_mean = colMeans(AIC_table)
bic_mean = colMeans(BIC_table)
aic_median = apply(AIC_table,2,median)
bic_median = apply(BIC_table,2,median)

plot.mean.median <- function(mean, median, IC="AIC") {
  x = 1:length(mean)
  title = ifelse(IC=="AIC","AIC Mean and Median","BIC Mean and Median")
  par(mar = c(4, 4, 4, 4))
  plot(x, mean, type = "l", col = "blue", lwd = 2, main=title,
       xlab = "eGARCH", ylab = "Mean", ylim = range(mean, median), xaxt = "n")
  points(x, mean, col = "blue", pch = 16)
  par(new = TRUE)
  plot(x, median, type = "l", col = "red", lwd = 2, yaxt = "n", ylab = "")
  lines(x, median, col = "red")
  points(x, median, col = "red", pch = 16)
  
  axis(side = 4, at = pretty(range(median)), labels = pretty(range(median)), col = "red")
  mtext("Median", side = 4, line = 3)
  legend("topleft", legend = c("Mean", "Median"), col = c("blue", "red"), lwd = 2, pch = 16)
  axis(1, at = x, labels = list[-20])
}

plot.mean.median(aic_mean, aic_median, IC="AIC")
plot.mean.median(bic_mean, bic_median, IC="BIC")


