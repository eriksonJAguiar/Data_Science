library(ggplot2)
library(ggcorrplot)
library(moments)

#--------- lendo a base--------
datas <- read.csv(file="~/Desktop/house-prices-advanced-regression-techniques/train.csv", header=TRUE, sep=",")

#extra as variaveis simbolicas
datas_num <- datas[sapply(datas, is.numeric)] 

#extra as variaveis simbolicas
data_simb <- datas[sapply(datas, is.object)] 

datas_num <- datas_num[ ,-1]


#------ Calculando a correlação ------
#calcula a correlacao de todos
corl_ger <- signif(cor(datas_num, method = "pearson"), 3)

#plot o heatmap da correlacao geral
ggcorrplot(corl_ger, type = "lower")

#atributos com maior correlacao: GrLivArea, GarageCars, GarageArea, TotalBsmtSF, X1stFlrSF -> Quantitativos
import_vars <- data.frame(datas_num$OverallQual,datas_num$GrLivArea, datas_num$GarageCars, datas_num$GarageArea, datas_num$TotalBsmtSF, datas_num$X1stFlrSF, datas_num$YearBuilt, datas_num$YearRemodAdd, datas_num$FullBath, datas_num$TotRmsAbvGrd ,datas_num$SalePrice)
names(import_vars) <- c("OverallQual","GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "XstFlrSF", "YearBuilt", "YearRemodAdd", "FullBath", "TotRmsAbvGrd", "SalePrice")

correlacao <- signif(cor(import_vars),3)

#plot o heatmap da correlacao
ggcorrplot(correlacao, type = "lower", lab=TRUE)
ggsave("~/Desktop/house-prices-advanced-regression-techniques/heatmap_Quant.jpg")

#Aplicando testes quantitativos----------


#metricas para OverallQual
print('-----OverallQual-------')
ov <- c(median(datas_num$OverallQual),mean(datas_num$OverallQual),sd(datas_num$OverallQual),max(datas_num$OverallQual),min(datas_num$OverallQual),skewness(datas_num$OverallQual),kurtosis(datas_num$OverallQual))
ggplot(datas_num, aes(x=datas_num$OverallQual)) + geom_histogram(binwidth = density(datas_num$OverallQual)$bw, color="black", fill="red") + 
  xlab(expression(bold('OverallQual'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_OverallQual.jpg")


#metricas para GrLivArea
print('-----GrLivArea-------')
gr <- c(median(datas_num$GrLivArea),mean(datas_num$GrLivArea),sd(datas_num$GrLivArea),max(datas_num$GrLivArea),min(datas_num$GrLivArea),skewness(datas_num$GrLivArea),kurtosis(datas_num$GrLivArea))
ggplot(datas_num, aes(x=datas_num$GarageCars)) + geom_histogram(aes(y=..density.., fill=..count..), color="black", fill="red") + 
  geom_density(colour = 'blue')+xlab(expression(bold('GrLivArea'))) + ylab(expression(bold('Frequences')))
ggsave("~~/Desktop/house-prices-advanced-regression-techniques/hist_GrLivArea.jpg")


#metricas para GarageCars
print('-----GarageCars-------')
grc <- c(median(datas_num$GarageCars),mean(datas_num$GarageCars),sd(datas_num$GarageCars),max(datas_num$GarageCars),min(datas_num$GarageCars),skewness(datas_num$GarageCars),kurtosis(datas_num$GarageCars))
ggplot(datas_num, aes(x=datas_num$GarageCars)) + geom_histogram(aes(y=..density.., fill=..count..), color="black", fill="red") + 
  geom_density(colour = 'blue')+xlab(expression(bold('GarageCars'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_GarageCars.jpg")


#metricas para GarageArea
print('-----GarageArea-------')
gra <- c(median(datas_num$GarageArea),mean(datas_num$GarageArea),sd(datas_num$GarageArea),max(datas_num$GarageArea),min(datas_num$GarageArea),skewness(datas_num$GarageArea),kurtosis(datas_num$GarageArea))
ggplot(datas_num, aes(x=datas_num$GarageArea)) + geom_histogram(aes(y=..density.., fill=..count..), color="black", fill="red") + 
  geom_density(colour = 'blue')+xlab(expression(bold('GarageArea'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_GarageCars.jpg")

#metricas para TotalBsmtSF
print('-----TotalBsmtSF-------')
tbsf <- c(median(datas_num$TotalBsmtSF),mean(datas_num$TotalBsmtSF),max(datas_num$TotalBsmtSF),min(datas_num$TotalBsmtSF),skewness(datas_num$TotalBsmtSF),kurtosis(datas_num$TotalBsmtSF))
ggplot(datas_num, aes(x=datas_num$TotalBsmtSF)) + geom_histogram(aes(y=..density.., fill=..count..), color="black", fill="red") + 
  geom_density(colour = 'blue')+xlab(expression(bold('TotalBsmtSF'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_TotalBsmtSF.jpg")



#metricas para X1stFlrSF
print('-----1stFlrSF-------')
stf <- c(median(datas_num$X1stFlrSF),mean(datas_num$X1stFlrSF),sd(datas_num$X1stFlrSF),max(datas_num$X1stFlrSF),min(datas_num$X1stFlrSF),skewness(datas_num$X1stFlrSF),kurtosis(datas_num$X1stFlrSF))
ggplot(datas_num, aes(x=datas_num$X1stFlrSF)) + geom_histogram(aes(y=..density.., fill=..count..), color="black", fill="red") + 
  geom_density(colour = 'blue')+xlab(expression(bold('X1stFlrSF'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_1stFlrSF.jpg")

#metricas para YearBuilt
print('-----YearBuilt-------')
yb <- c(median(datas_num$YearBuilt),mean(datas_num$YearBuilt),sd(datas_num$YearBuilt),max(datas_num$YearBuilt),min(datas_num$YearBuilt),skewness(datas_num$YearBuilt),kurtosis(datas_num$YearBuilt))
ggplot(datas_num, aes(x=datas_num$YearBuilt)) + geom_histogram(binwidth = density(datas_num$YearBuilt)$bw, color="black", fill="red") + 
  xlab(expression(bold('YearBuilt'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_YearBuilt.jpg")

#metricas para YearRemodAdd
print('-----YearRemodAdd-------')
ya <- c(median(datas_num$YearRemodAdd),mean(datas_num$YearRemodAdd),sd(datas_num$YearRemodAdd),max(datas_num$YearRemodAdd),min(datas_num$YearRemodAdd),skewness(datas_num$YearRemodAdd),kurtosis(datas_num$YearRemodAdd))
ggplot(datas_num, aes(x=datas_num$YearRemodAdd)) + geom_histogram(binwidth = density(datas_num$YearRemodAdd)$bw, color="black", fill="red") + 
  xlab(expression(bold('YearRemodAdd'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_YearRemodAdd.jpg")

#metricas para FullBath
print('-----FullBath-------')
fb <- c(median(datas_num$FullBath),mean(datas_num$FullBath),sd(datas_num$FullBath),max(datas_num$FullBath),min(datas_num$FullBath),skewness(datas_num$FullBath),kurtosis(datas_num$FullBath))
ggplot(datas_num, aes(x=datas_num$FullBath)) + geom_histogram(binwidth = density(datas_num$FullBath)$bw, color="black", fill="red") + 
  xlab(expression(bold('FullBath'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_FullBath.jpg")

#metricas para TotRmsAbvGrd
print('-----TotRmsAbvGrd-------')
rgd <- c(median(datas_num$TotRmsAbvGrd),mean(datas_num$TotRmsAbvGrd),sd(datas_num$TotRmsAbvGrd),max(datas_num$TotRmsAbvGrd),min(datas_num$TotRmsAbvGrd),skewness(datas_num$TotRmsAbvGrd),kurtosis(datas_num$TotRmsAbvGrd))
ggplot(datas_num, aes(x=datas_num$TotRmsAbvGrd)) + geom_histogram(binwidth = density(datas_num$TotRmsAbvGrd)$bw, color="black", fill="red") + 
  xlab(expression(bold('TotRmsAbvGrd'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_TotRmsAbvGrd.jpg")


#metricas para SalePrice
print('-----SalePrice-------')
sale <- c(median(datas_num$SalePrice),mean(datas_num$SalePrice),sd(datas_num$SalePrice),max(datas_num$SalePrice),min(datas_num$SalePrice),skewness(datas_num$SalePrice),kurtosis(datas_num$SalePrice))
ggplot(datas_num, aes(x=datas_num$SalePrice)) + geom_histogram(aes(y=..density.., fill=..count..), color="black", fill="red") + 
  geom_density(colour = 'blue')+xlab(expression(bold('SalePrices'))) + ylab(expression(bold('Frequences')))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/hist_SalePrices.jpg")


#juntandos dados
met <- cbind(ov,gr,grc,gra,tbsf,stf,yb,ya,fb,rgd,sale)
metrics <- data.frame(met)
colnames(met) <- c("OverallQua","GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "YearBuilt", "YearRemodAdd", "FullBath", "TotRmsAbvGrd", "SalePrice")
rownames(metrics) <- c("median","mean","standard_deviation","max","min","skewness","kurtosis")
write.csv(metrics, file = "~/Desktop/house-prices-advanced-regression-techniques/metrics.csv", sep=";")

#gerando  boxplot para cada variavel
ggplot(datas_num, aes(x = 1, y = datas_num$SalePrice)) + geom_boxplot() + xlab(expression(bold("OverallQual"))) + ylab(expression(bold("SalePrices")))
ggsave("~/Desktop/house-prices-advanced-regression-techniques/boxplot_SalePrices_OverallQual.jpg")

#------------Qualitativas-------------------


#-----------Simbolic-----------------------
#Plot gráficos de barra para representar valor qualitativos
jpeg(file="~/Desktop/house-prices-advanced-regression-techniques/barra_Foundation.jpg")
plot(data_simb$Foundation, col='red')
dev.off()
jpeg(file="~/Desktop/house-prices-advanced-regression-techniques/barra_Neighborhood.jpg")
plot(data_simb$Neighborhood, col='red')
dev.off()
jpeg(file="~/Desktop/house-prices-advanced-regression-techniques/barra_GarageType.jpg")
plot(data_simb$GarageType, col='red')
dev.off()
jpeg(file="~/Desktop/house-prices-advanced-regression-techniques/barra_KitchenQual.jpg")
plot(data_simb$KitchenQual, col='red')
dev.off()    




