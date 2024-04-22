# INPUT:
# 1. Training pairs_spread (csv): index = Date, 73 pairs
# 2. Training half_life (csv): Rolling window, index = pairs
# 3. Testing pairs_spread (csv): index=Date, 73 pairs
# 4. Testing half_life (csv): Rolling window, index = pairs
# 5. Pair_description (csv): Tickers corresponding to the pair
# OUTPUT:
# 1. Training spread scaled (csv)
# 2. Training volatility (csv)
# 3. Training z_score (csv)
# 4. Testing spread scaled (csv)
# 5. Testing volatility (csv)
# 6. Testing z_score (csv)
# 7. Heatmap image

# Pairs information ------
pair_description = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\pair_df.csv", row.names=1)
pair_description$pair_name = paste(pair_description$Ticker.1, pair_description$Ticker.2)
pairs.name = pair_description$pair_name
pairs.no = length(pair_description$pair_name); pairs.no

# Training spreads pre-processing ------
## First date: 1/30/2012, Last date: 8/5/2021
library(xts)
train.pair_spread = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\train_pair_spread.csv", row.names=1)
train.pair_spread = as.xts(train.pair_spread) # Convert spreads to time series
train.half_life = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\train_half_life.csv", row.names=1)
for (col in 1:pairs.no){
  window = train.half_life$Half.Life[col]
  train.pair_spread[,col] = rollmean(train.pair_spread[,col], k = window, align = "right", fill = NA)
} # Convert to rolling spreads based on half_life
train.spread_scaled = scale(na.omit(train.pair_spread), center = TRUE, scale = TRUE) # Standardization for DCC_GARCH training
train.length = nrow(train.spread_scaled); train.length

# DCC_GARCH Training ----
# eGARCH follows a t-dist, DCC_GARCH follows multi-variate t-dist
library(rmgarch)
garch.spec = ugarchspec(variance.model = list(model="eGARCH", garchOrder=c(1,2)),
                        mean.model = list(armaOrder=c(1,2),include.mean=FALSE),distribution.model = "std")
dcc_spec = dccspec(uspec = multispec(replicate(garch.spec, n=pairs.no)),
                   dccOrder = c(1,1), model="DCC", distribution = "mvt")
dcc_GARCH = dccfit(spec = dcc_spec, data = train.spread_scaled) # Takes around 2h to run

# train.vol -------
train.cov_matrix = vector("list",train.length)
train.vol = matrix(NA, nrow = train.length, ncol = pairs.no,
                   dimnames = list(gsub("CST", "America/Chicago",index(train.spread_scaled)), pairs.name))
for (day in (1:train.length)){
  train.cov_matrix[[day]] = as.matrix(as.data.frame(dcc_GARCH@mfit$Q[day]))
  train.vol[day,] = diag(train.cov_matrix[[day]])
}
## train.vol plot -----
par(mfrow=c(4,4))
for (pair in 1:pairs.no){
  vol = na.omit(train.vol[,pairs.name[pair]])
  ts.plot(vol, main = pairs.name[pair], ylab = "Training Volatility")
}

# train.z_score -----
train.z_score = train.spread_scaled / train.vol
colnames(train.z_score) = pairs.name
## train.z_score plot -----
par(mfrow=c(3,3))
for (pair in 1:pairs.no){
  z_score = na.omit(train.z_score[,pairs.name[pair]])
  ts.plot(z_score, main = pairs.name[pair], ylab = "Training Z score")
}
# Write training csv -------
write.zoo(as.zoo(train.spread_scaled), file="train_spread_scaled.csv", sep=",")
write.zoo(as.zoo(as.xts(train.vol)), file="train_spread_vol.csv", sep=",")
write.zoo(as.zoo(train.z_score), file="train_z_score.csv", sep=",")

# DCC_GARCH Result ------
theta1 = tail(dcc_GARCH@mfit$matcoef,3)[1,1] # DCC theta1
theta2 = tail(dcc_GARCH@mfit$matcoef,2)[1,1] # DCC theta2
latest.Q = as.data.frame(tail(dcc_GARCH@mfit$Q,1))
one.third = (train.length%/%3) # Burn-in first 1/3 data
Q_sum = as.matrix(as.data.frame(dcc_GARCH@mfit$Q[one.third]))
for (day in (one.third:train.length)){Q_sum = Q_sum + as.matrix(as.data.frame(train.cov_matrix[day]))}
Q_mean = Q_sum / length(one.third:train.length)
## Heatmap -----
jpeg(file="heatmap.jpg", quality=90)
heatmap(Q_mean, col = colorRampPalette(c("green3", "yellow2", "red3"))(100), 
        main = "Covariance Heatmap", xlab = "Pairs", ylab = "Pairs")
dev.off()

# testing spreads pre-processing ----
## First date: 8/23/2012, Last date: 12/29/2023
test.pair_spread = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\test_pair_spread.csv", row.names=1)
test.pair_spread = as.xts(test.pair_spread)
test.half_life = read.csv("C:\\Users\\user\\Desktop\\Year 3 Sem 2\\FINA 4380\\test_half_life.csv", row.names=1)
for (col in 1:pairs.no){
  window = test.half_life$Half.Life[col]
  test.pair_spread[,col] = rollmean(test.pair_spread[,col], k = window, align = "right", fill = NA)
}
test.spread_scaled = scale(na.omit(test.pair_spread), center = TRUE, scale = TRUE)
latest.eta = as.numeric(head(test.spread_scaled,1)) # First data is used as input of DCC_GARCH
test.spread_scaled = test.spread_scaled[-1,]
test.length = nrow(test.spread_scaled); test.length

# test.vol -----
test.cov_matrix = vector("list",test.length)
test.vol = matrix(NA, nrow = test.length, ncol = pairs.no,
                  dimnames = list(gsub("CST", "America/Chicago",index(test.spread_scaled)), pairs.name))
Q.t1 = (1-theta1-theta2)*Q_mean + theta2 * (latest.eta %*% t(latest.eta)) + theta1 * latest.Q
test.cov_matrix[[1]] = as.matrix(Q.t1)
test.vol[1,] = diag(as.matrix(Q.t1))
for (row in 2:test.length){
  if (row == 2){Q.lag = as.matrix(Q.t1)}
  eta.lag = as.numeric(test.spread_scaled[row,])
  Q.next = (1-theta1-theta2)*Q_mean + theta2 * (eta.lag %*% t(eta.lag)) + theta1 * Q.lag
  test.cov_matrix[[row]] = as.matrix(Q.next)
  test.vol[row,] = diag(test.cov_matrix[[row]])
  Q.lag = Q.next
}
## test.vol plot -----
par(mfrow=c(4,4))
for (pair in 1:pairs.no){
  vol = na.omit(test.vol[,pairs.name[pair]])
  ts.plot(vol, main = pairs.name[pair], ylab = "Testing Volatility")
}

# test.z_score -----
test.z_score = test.spread_scaled / test.vol
colnames(test.z_score) = pairs.name
## test.z_score plot -----
par(mfrow=c(3,3))
for (pair in 1:pairs.no){
  z_score = na.omit(test.z_score[,pairs.name[pair]])
  ts.plot(z_score, main = pairs.name[pair], ylab = "Testing Z score")
}

# Write training csv -------
write.zoo(as.zoo(test.spread_scaled), file="test_spread_scaled.csv", sep=",")
write.zoo(as.zoo(as.xts(test.vol)), file="test_spread_vol.csv", sep=",")
write.zoo(as.zoo(test.z_score), file="test_z_score.csv", sep=",")
