rm(list=ls())
cat("\014")
library(wooldridge)
library(ISLR)
library(leaps)
library(stats)
library(ggcorrplot)
library(RColorBrewer)
library(tidyverse)
library(glmnet)
library(pls)
library(caret)
library(MLmetrics)
library(car)
library(selectiveInference)
library(covTest)
library(hdm)
attach(discrim)

# Variable independiente psoda

set.seed(44)

##### a)

discrim <-na.omit(discrim)

#Eliminamos las vaiables chain y state porque ya estan en forma binaria en otras variables.

#Eliminamos las variables hrsopen2, pentree2, wagest2, nregs2, psoda2, pfries2, nmgrs2, emp2 por estar medidas en otra fecha.
discrim <- dplyr::select(discrim, -c(state, chain, lpsoda, hrsopen2, pentree2, wagest2, nregs2, psoda2, pfries2, nmgrs2, emp2))

#Para hacer la correlación, quitamos las variables compown, NJ, BK, KFC, RR, lpfries, lhseval, lincome y ldensity

datos_cor <- dplyr::select(discrim, -c(compown, county, NJ, BK, KFC, RR, lpfries, lhseval, lincome, ldensity))

ggcorrplot(cor(datos_cor), hc.order = TRUE, type = "lower",
           lab = TRUE, lab_size = 1.5)

# Quitamos las variables  prppov, hseval, prpncar por tener alta correlación 

# Eliminamos la variable county ya que es una etiqueta con 310 niveles.

discrim <- dplyr::select(discrim, -c(prppov, county, hseval, prpncar,lhseval))

# Seleccionamos muestra de entrenamiento y test
muestra <- sample(1:nrow(discrim), round(0.79*nrow(discrim), 0))
entrenamiento <- discrim[muestra, ]
test <- discrim[-muestra, ]

reg_1 = lm(psoda ~ ., 
           data = entrenamiento)
summary(reg_1)

pred = predict(reg_1, newdata = test)
error_reg_1 <- sqrt(MSE(y_pred = pred,y_true = test$psoda))
error_reg_1

##### b)

#Esta es la funcion para predecir los regsubsets 

predict.regsubsets=function(object, newdata, id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form, newdata)
  coefi=coef(object, id=id)
  mat[, names(coefi)]%*%coefi
}

k = 10
folds = sample(1:k,nrow(entrenamiento),replace=TRUE)

cv_error_sub_10=matrix(NA,k,(ncol(entrenamiento)-1), dimnames=list(NULL, paste(1:(ncol(entrenamiento)-1))))

for(j in 1:k){
  reg_full_10 = regsubsets(psoda ~., data=entrenamiento[folds != j,], nvmax = (ncol(entrenamiento)-1))
  
  for (i in 1:(ncol(entrenamiento)-1)){
    pred = predict.regsubsets(reg_full_10, entrenamiento[folds == j, ], id = i)
    cv_error_sub_10[j, i] = mean((entrenamiento$psoda[folds == j] - pred)^2)
  }
}

cv_error_media_10 = apply(cv_error_sub_10 ,2,mean)
cv_error_media_10

plot(cv_error_media_10,pch=19,type="b", xlab="Numero de variables", ylab="Error CV")
points(which.min(cv_error_media_10),cv_error_media_10[which.min(cv_error_media_10)], col="red",cex=2,pch=18)

mejor_reg=regsubsets (psoda~.,data=entrenamiento , nvmax=(ncol(entrenamiento)-1))
coef(mejor_reg ,which.min(cv_error_media_10))

reg_sub_10 =regsubsets(psoda~.,data= entrenamiento,nvmax=(ncol(entrenamiento)-1))

pred_reg_sub_10 = predict.regsubsets(reg_sub_10, newdata = test, id=which.min(cv_error_media_10))
error_mss_sub_10 <- sqrt(mean((test$psoda - pred_reg_sub_10)^2))
error_mss_sub_10

# Regla codo
codo_sub_10 = sd(cv_error_media_10)
which.max(cv_error_media_10 - codo_sub_10 <= min(cv_error_media_10))

reg_sub_10_codo =regsubsets(psoda~.,data= entrenamiento,nvmax=(ncol(entrenamiento)-1))

pred_reg_sub_10_codo = predict.regsubsets(reg_sub_10_codo, newdata = test, id=which.max(cv_error_media_10 - codo_sub_10 <= min(cv_error_media_10)))
error_mss_sub_10_codo <- sqrt(mean((test$psoda - pred_reg_sub_10_codo)^2))
error_mss_sub_10_codo

##### c)

cv_error_sub_for_10=matrix(NA,k,(ncol(entrenamiento)-1), 
                           dimnames=list(NULL, paste(1:(ncol(entrenamiento)-1))))

for(j in 1:k){
  reg_for_10=regsubsets(psoda~.,data=entrenamiento[folds != j,], nvmax= (ncol(entrenamiento)-1),
                        method= "forward")
  
  for (i in 1:(ncol(entrenamiento)-1)){
    pred = predict.regsubsets(reg_for_10, entrenamiento[folds == j, ], id = i)
    cv_error_sub_for_10[j, i] = mean((entrenamiento$psoda[folds == j] - pred)^2)
  }
}

cv_error_for_media_10 = apply(cv_error_sub_for_10 ,2,mean)
cv_error_for_media_10

plot(cv_error_for_media_10,pch=19,type="b", xlab="Numero de variables", ylab="Error CV")
points(which.min(cv_error_for_media_10),cv_error_for_media_10[which.min(cv_error_for_media_10)], col="red",cex=2,pch=18)

mejor_reg=regsubsets (psoda~.,data=entrenamiento , nvmax=(ncol(entrenamiento)-1), method= "forward")
coef(mejor_reg ,which.min(cv_error_for_media_10))

reg_for_10 =regsubsets(psoda~.,data= entrenamiento,nvmax=(ncol(entrenamiento)-1), method= "forward")

pred_reg_for_10 = predict.regsubsets(reg_for_10, newdata = test, id=which.min(cv_error_for_media_10))
error_mss_for_10 <- sqrt(mean((test$psoda - pred_reg_for_10)^2))
error_mss_for_10

# Regla codo
codo_for_10 = sd(cv_error_for_media_10)
which.max(cv_error_for_media_10 - codo_for_10 <= min(cv_error_for_media_10))

reg_for_10_codo =regsubsets(psoda~.,data= entrenamiento,nvmax=(ncol(entrenamiento)-1))

pred_reg_for_10_codo = predict.regsubsets(reg_for_10_codo, newdata = test, id=which.max(cv_error_for_media_10 - codo_for_10 <= min(cv_error_for_media_10)))
error_mss_for_10_codo <- sqrt(mean((test$psoda - pred_reg_for_10_codo)^2))
error_mss_for_10_codo

##### d) 

#Mejor selección de conjuntos

k = 5
folds = sample(1:k,nrow(entrenamiento),replace=TRUE)

cv_error_sub_5=matrix(NA,k,(ncol(entrenamiento)-1), dimnames=list(NULL, paste(1:(ncol(entrenamiento)-1))))

for(j in 1:k){
  reg_full_5 = regsubsets(psoda ~., data=entrenamiento[folds != j,], nvmax = (ncol(entrenamiento)-1))
  
  for (i in 1:(ncol(entrenamiento)-1)){
    pred = predict.regsubsets(reg_full_5, entrenamiento[folds == j, ], id = i)
    cv_error_sub_5[j, i] = mean((entrenamiento$psoda[folds == j] - pred)^2)
  }
}

cv_error_media_5 = apply(cv_error_sub_5 ,2,mean)
cv_error_media_5

plot(cv_error_media_5,pch=19,type="b", xlab="Numero de variables", ylab="Error CV")
points(which.min(cv_error_media_5),cv_error_media_5[which.min(cv_error_media_5)], col="red",cex=2,pch=18)

mejor_reg=regsubsets (psoda~.,data=entrenamiento , nvmax=(ncol(entrenamiento)-1))
coef(mejor_reg ,which.min(cv_error_media_5))

reg_sub_5 =regsubsets(psoda~.,data= entrenamiento,nvmax=(ncol(entrenamiento)-1))

pred_reg_sub_5 = predict.regsubsets(reg_sub_5, newdata = test, id=which.min(cv_error_media_5))
error_mss_sub_5 <- sqrt(mean((test$psoda - pred_reg_sub_5)^2))
error_mss_sub_5

# Regla codo
codo_sub_5 = sd(cv_error_media_5)
which.max(cv_error_media_5 - codo_sub_5 <= min(cv_error_media_5))

#Selección por pasos hacia adelante

cv_error_sub_for_5=matrix(NA,k,(ncol(entrenamiento)-1), dimnames=list(NULL, paste(1:(ncol(entrenamiento)-1))))

for(j in 1:k){
  reg_for_5=regsubsets(psoda~.,data=entrenamiento[folds != j,], nvmax= (ncol(entrenamiento)-1),method= "forward")
  
  for (i in 1:(ncol(entrenamiento)-1)){
    pred = predict.regsubsets(reg_for_5, entrenamiento[folds == j, ], id = i)
    cv_error_sub_for_5[j, i] = mean((entrenamiento$psoda[folds == j] - pred)^2)
  }
}

cv_error_for_media_5 = apply(cv_error_sub_for_5 ,2,mean)
cv_error_for_media_5

plot(cv_error_for_media_5,pch=19,type="b", xlab="Numero de variables", ylab="Error CV")
points(which.min(cv_error_for_media_5),cv_error_for_media_5[which.min(cv_error_for_media_5)], col="red",cex=2,pch=18)

mejor_reg=regsubsets (psoda~.,data=entrenamiento , nvmax=(ncol(entrenamiento)-1), method= "forward")
coef(mejor_reg ,which.min(cv_error_for_media_5))

reg_for_5 =regsubsets(psoda~.,data= entrenamiento,nvmax=(ncol(entrenamiento)-1), method= "forward")

pred_reg_for_5 = predict.regsubsets(reg_for_5, newdata = test, id=which.min(cv_error_for_media_5))
error_mss_for_5 <- sqrt(mean((test$psoda - pred_reg_for_5)^2))
error_mss_for_5

# Regla codo

codo_for_5 = sd(cv_error_for_media_5)
which.max(cv_error_for_media_5 - codo_for_5 <= min(cv_error_for_media_5))

reg_for_5_codo =regsubsets(psoda~.,data= entrenamiento,nvmax=(ncol(entrenamiento)-1))

pred_reg_for_5_codo = predict.regsubsets(reg_for_5_codo, newdata = test, id=which.max(cv_error_for_media_5 - codo_for_5 <= min(cv_error_for_media_5)))
error_mss_for_5_codo <- sqrt(mean((test$psoda - pred_reg_for_5_codo)^2))
error_mss_for_5_codo

##### e)

tabla <- data.frame("Regresión" = c("Minimos cuadrados ordinarios",
                                    "Seleccion de subconjuntos CV 10",
                                    "Seleccion por pasos hacia adelante CV 10",
                                    "Seleccion de subconjuntos CV 5", 
                                    "Seleccion por pasos hacia adelante CV 5" ),
                    "Error_prueba" = c(error_reg_1,
                                       error_mss_sub_10, 
                                       error_mss_for_10,
                                       error_mss_sub_5,
                                       error_mss_for_5_codo),
                    "Número_variables" = c(" ",which.min(cv_error_media_10),
                                              which.min(cv_error_for_media_10),
                                              which.min(cv_error_media_5),
                                              which.min(cv_error_for_media_5)))

tabla

##### f)
# Lo miramos por Bonferroni-Holm

reg_lineal <- lm(psoda ~pfries + pentree +prpblck +NJ+ BK + RR, data= entrenamiento)
summary(reg_lineal)

p = c(3.75*10^(-16), 0.000513, 0.017370, 0.000329, 1.11*10^(-8), 6.72*10^(-8)) #p-valores de la regresion

# Los que sean TRUE los seleccionamos.

p <= 0.05/length(p)

# Eliminamos la variable prpblack

reg_nueva <- lm(psoda ~pfries + pentree +NJ+ BK + RR, data= entrenamiento)

pred = predict(reg_nueva, newdata = test)
error_reg_nueva <- sqrt(MSE(y_pred = pred,y_true = test$psoda))
error_reg_nueva

# El error es mayor que antes

##### g)
matriz_entrenamiento = model.matrix(psoda~., data=entrenamiento)[,-1]
matriz_test = model.matrix(psoda~., data=test)[,-1]
grid = 10^seq(4, -2, length=100)
modelo_ridge_10 = cv.glmnet(matriz_entrenamiento, entrenamiento$psoda, alpha=0, lambda=grid,  nfolds = 10)
mejor_lambda_ridge_10 = modelo_ridge_10$lambda.min
mejor_lambda_ridge_10

plot(modelo_ridge_10)
modelo_ridge_10_l=glmnet(matriz_entrenamiento,entrenamiento$psoda,alpha=0,lambda=grid, thresh = 1e-12)
prediccion_ridge_10 = predict(modelo_ridge_10_l, newx=matriz_test, s=mejor_lambda_ridge_10)
a = sqrt(mean((test$psoda - prediccion_ridge_10)^2))
a

# Regla del codo
lambda_codo_ridge_10 <- modelo_ridge_10$lambda.1se
lambda_codo_ridge_10

prediccion_ridge_10_2=predict(modelo_ridge_10_l,s=lambda_codo_ridge_10,newx=matriz_test)
error.ridge.2 <- sqrt(mean((prediccion_ridge_10_2-test$psoda )^2))
error.ridge.2


##### h) 

modelo_LASSO_10= cv.glmnet(matriz_entrenamiento, entrenamiento$psoda, alpha=1, lambda=grid, nfolds = 10)
mejor_lambda_LASSO_10 = modelo_LASSO_10$lambda.min
mejor_lambda_LASSO_10

plot(modelo_LASSO_10)
modelo_LASSO_10_l=glmnet(matriz_entrenamiento,entrenamiento$psoda,alpha=1,lambda=grid, thresh = 1e-12)
prediccion_LASSO_10 = predict(modelo_LASSO_10_l, newx=matriz_test, s=mejor_lambda_LASSO_10)
b = sqrt(mean((test$psoda - prediccion_LASSO_10)^2))
b

# Regla del codo
lambda_codo_LASSO_10 <- modelo_LASSO_10$lambda.1se
lambda_codo_LASSO_10

prediccion_LASSO_10_2=predict(modelo_LASSO_10_l,s=lambda_codo_LASSO_10,newx=matriz_test)
error.LASSO.2 <- sqrt(mean((prediccion_LASSO_10_2-test$psoda )^2))
error.LASSO.2

##### i) 

# Ridge CV-5

modelo_ridge_5 = cv.glmnet(matriz_entrenamiento, entrenamiento$psoda, alpha=0, lambda=grid, nfolds = 5)
mejor_lambda_ridge_5 = modelo_ridge_5$lambda.min
mejor_lambda_ridge_5

modelo_ridge_5_l=glmnet(matriz_entrenamiento,entrenamiento$psoda,alpha=0,lambda=grid, thresh = 1e-12)
prediccion_ridge_5 = predict(modelo_ridge_5_l, newx=matriz_test, s=mejor_lambda_ridge_5)
c =sqrt(mean((test$psoda - prediccion_ridge_5)^2))
c

# Regla del codo

lambda_codo_ridge_5 <- modelo_ridge_5$lambda.1se
lambda_codo_ridge_5

prediccion_ridge_5_2=predict(modelo_ridge_5_l,s=lambda_codo_ridge_5,newx=matriz_test)
error.ridge.2 <- sqrt(mean((prediccion_ridge_5_2-test$psoda )^2))
error.ridge.2

# LASSO CV-5

modelo_LASSO_5= cv.glmnet(matriz_entrenamiento, entrenamiento$psoda, alpha=1, lambda=grid, nfolds = 5)
mejor_lambda_LASSO_5 = modelo_LASSO_5$lambda.min
mejor_lambda_LASSO_5

modelo_LASSO_5_l=glmnet(matriz_entrenamiento,entrenamiento$psoda,alpha=1,lambda=grid, thresh = 1e-12)
prediccion_LASSO_5 = predict(modelo_LASSO_5_l, newx=matriz_test, s=mejor_lambda_LASSO_5)
d =sqrt(mean((test$psoda - prediccion_LASSO_5)^2))
d

# Regla del codo
lambda_codo_LASSO_5 <- modelo_LASSO_5$lambda.1se
lambda_codo_LASSO_5

prediccion_LASSO_5_2=predict(modelo_LASSO_5_l,s=lambda_codo_LASSO_5,newx=matriz_test)
error.LASSO.2 <- sqrt(mean((prediccion_LASSO_5_2-test$psoda )^2))
error.LASSO.2

##### j)

acp=pcr(psoda~., data=entrenamiento,scale=TRUE, validation="CV")

# CV_10
acp_cv_10 <- crossval(acp, segments = 10)
summary(acp_cv_10, what = "validation")

acp_pred_10_cv=predict(acp,newdata=test,ncomp=13) 
error_acp_10_cv<- sqrt(mean((acp_pred_10_cv - test$psoda)^2))
error_acp_10_cv

# Regla del codo
regla_codo_10 <- selectNcomp(acp, method = "onesigma", plot = TRUE, validation = "CV",
                            segments = 10)
regla_codo_10
acp_pred_10_codo=predict(acp,newdata=test,ncomp=regla_codo_10)
error_acp_10_codo <- sqrt(mean((acp_pred_10_codo - test$psoda)^2))
error_acp_10_codo

#CV_5
acp_cv_5 <- crossval(acp, segments = 5)
summary(acp_cv_5, what = "validation")

acp_pred_5_cv=predict(acp,newdata=test,ncomp=13) 
error_acp_5_cv<- sqrt(mean((acp_pred_5_cv - test$psoda)^2))
error_acp_5_cv

# Regla del codo
regla_codo_5 <- selectNcomp(acp, method = "onesigma", plot = TRUE, validation = "CV",
                          segments = 5)
regla_codo_5
acp_pred_5_codo=predict(acp,newdata=test,ncomp=regla_codo_5)
error_acp_5_codo <- sqrt(mean((acp_pred_5_codo - test$psoda)^2))
error_acp_5_codo

##### k) 

pls=plsr(psoda~., data=entrenamiento ,scale=TRUE, validation="CV")

# PLS CV 10

pls_cv_10 <- crossval(pls, segments = 10)
summary(pls_cv_10, what = "validation")

pls_pred_10_cv=predict(pls,newdata=matriz_test,ncomp=3)
error_pls_10 <- sqrt(mean((pls_pred_10_cv - test$psoda)^2))
error_pls_10

codo_pls_10 <- selectNcomp(pls, method = "onesigma", plot = TRUE, validation = "CV",
                            segments = 10)
codo_pls_10
pls_pred_10_codo=predict(pls,newdata=matriz_test,ncomp=codo_pls_10)
error_pls_10_codo <- sqrt(mean((pls_pred_10_codo - test$psoda)^2))
error_pls_10_codo

# PLS CV 5

pls_cv_5 <- crossval(pls, segments = 5)
plot(RMSEP(pls_cv_5), legendpos="topright")
summary(pls_cv_5, what = "validation")

pls_pred_5_cv=predict(pls,newdata=matriz_test,ncomp=2)
error_pls_5 <- sqrt(mean((pls_pred_5_cv - test$psoda)^2))
error_pls_5

codo_pls_5 <- selectNcomp(pls, method = "onesigma", plot = TRUE, validation = "CV",
                           segments = 5)
codo_pls_5
pls_pred_5_codo=predict(pls,newdata=matriz_test,ncomp=codo_pls_5)
error_pls_5_codo <- sqrt(mean((pls_pred_5_codo - test$psoda)^2))
error_pls_5_codo

# Randomization

codo_pls_random <- selectNcomp(pls, method = "randomization", plot = TRUE)
codo_pls_random
pls_pred_random <- predict(pls,newdata=matriz_test,ncomp=codo_pls_random)
error_pls_random <- sqrt(mean((pls_pred_random - test$psoda)^2))
error_pls_random

##### l)
lambda_grid <- 10^seq(2,-2, length = 100)
alpha_grid <- seq(0,1, by = 0.05)
Control <- trainControl(method = "cv", number = 10)
buscar_grid <- expand.grid(alpha = alpha_grid, lambda = lambda_grid)
entrenamiento_modelo <- train(psoda~., data = entrenamiento, method = "glmnet", 
                          tuneGrid = buscar_grid, trControl = Control,
                          tuneLength = 10,
                          standardize = TRUE, maxit = 1000000)

best_tune_EN_10 <- entrenamiento_modelo$bestTune
entrenamiento_modelo$bestTune
plot(entrenamiento_modelo)

modelo_glmnet <- entrenamiento_modelo$finalModel
coef(modelo_glmnet, s = entrenamiento_modelo$bestTune$lambda)
mejor_modelo <- glmnet(matriz_entrenamiento,entrenamiento$psoda, alpha=entrenamiento_modelo$bestTune$alpha,
                     lambda = entrenamiento_modelo$bestTune$lambda, thresh = 1e-12)
coef(mejor_modelo, s = entrenamiento_modelo$bestTune$lambda)
cbind(coef(mejor_modelo, s = entrenamiento_modelo$bestTune$lambda),
      coef(modelo_glmnet,
           s = entrenamiento_modelo$bestTune$lambda))

pred_LASSO_elastic_10 <- predict(mejor_modelo,s=entrenamiento_modelo$bestTune$lambda,newx=matriz_test)
error_pred_LASSO_elastic_10 <- sqrt(mean((pred_LASSO_elastic_10 - test$psoda)^2))
error_pred_LASSO_elastic_10

##### m) 

Control <- trainControl(method = "cv", number = 5)
buscar_grid <- expand.grid(alpha = alpha_grid, lambda = lambda_grid)
entrenamiento_modelo <- train(psoda~., data = entrenamiento, method = "glmnet", 
                              tuneGrid = buscar_grid, trControl = Control,
                              tuneLength = 10,
                              standardize = TRUE, maxit = 1000000)
best_tune_EN_5 <- entrenamiento_modelo$bestTune
entrenamiento_modelo$bestTune
plot(entrenamiento_modelo)

modelo_glmnet <- entrenamiento_modelo$finalModel
coef(modelo_glmnet, s = entrenamiento_modelo$bestTune$lambda)
mejor_modelo <- glmnet(matriz_entrenamiento,entrenamiento$psoda, alpha=entrenamiento_modelo$bestTune$alpha,
                       lambda = entrenamiento_modelo$bestTune$lambda, thresh = 1e-12)
coef(mejor_modelo, s = entrenamiento_modelo$bestTune$lambda)
cbind(coef(mejor_modelo, s = entrenamiento_modelo$bestTune$lambda),
      coef(modelo_glmnet,
           s = entrenamiento_modelo$bestTune$lambda))

pred_LASSO_elastic_5 <- predict(mejor_modelo,s=entrenamiento_modelo$bestTune$lambda,newx=matriz_test)
error_pred_LASSO_elastic_5 <- sqrt(mean((pred_LASSO_elastic_5 - test$psoda)^2))
error_pred_LASSO_elastic_5

##### n) 

# Ridge con cv10

n = nrow(matriz_entrenamiento)
beta_ridge_10 = coef(modelo_ridge_10_l, s=mejor_lambda_ridge_10/n, exact=TRUE, x = matriz_entrenamiento, y = entrenamiento$psoda)[-1]
out_ridge_10 = fixedLassoInf(matriz_entrenamiento,entrenamiento$psoda ,beta_ridge_10, mejor_lambda_ridge_10/n)
out_ridge_10

# Eliminar todas las variables menos la 17,18,20

matriz_entrenamiento_nueva <- matriz_entrenamiento[,c(17,18,20)]
matriz_test_nueva <- matriz_test[,c(17,18,20)]


grid=10^seq(4,-2, length =100)
cv.mod=cv.glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=0,lambda=grid, nfolds = 10)
plot(cv.mod)
mejorlambda_1=cv.mod$lambda.min
mejorlambda_1

mod=glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=0,lambda=grid)
pred=predict(mod,s=mejorlambda_1 ,newx=matriz_test_nueva)
error_1 <- sqrt(mean((pred-test$psoda )^2))
error_1



# LASSO con cv10

beta_LASSO_10 = coef(modelo_LASSO_10_l, s=mejor_lambda_LASSO_10/n, exact=TRUE, x = matriz_entrenamiento, y = entrenamiento$psoda)[-1]
out_LASSO_10 <- fixedLassoInf(matriz_entrenamiento,entrenamiento$psoda, beta_LASSO_10 ,mejor_lambda_LASSO_10/n)
out_LASSO_10

# Eliminar todas las variables menos la 17,18,20

grid=10^seq(4,-2, length =100)
cv.mod=cv.glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=1,lambda=grid, nfolds = 10)
plot(cv.mod)
mejorlambda_2=cv.mod$lambda.min
mejorlambda_2

mod=glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=1,lambda=grid)
pred=predict(mod,s=mejorlambda_2 ,newx=matriz_test_nueva)
error_2 <- sqrt(mean((pred-test$psoda )^2))
error_2

# Ridge con cv5

beta_ridge_5 = coef(modelo_ridge_5_l, s=mejor_lambda_ridge_5/n, exact=TRUE, x = matriz_entrenamiento, y = entrenamiento$psoda)[-1]
out_ridge_5 = fixedLassoInf(matriz_entrenamiento,entrenamiento$psoda ,beta_ridge_5, mejor_lambda_ridge_5/n)
out_ridge_5

# Eliminar todas las variables menos la 17,18,20

grid=10^seq(4,-2, length =100)
cv.mod=cv.glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=0,lambda=grid, nfolds = 5)
plot(cv.mod)
mejorlambda_3=cv.mod$lambda.min
mejorlambda_3

mod=glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=0,lambda=grid)
pred=predict(mod,s=mejorlambda_3 ,newx=matriz_test_nueva)
error_3 <- sqrt(mean((pred-test$psoda )^2))
error_3

# LASSO con cv5

beta_LASSO_5 = coef(modelo_LASSO_5_l, s=mejor_lambda_LASSO_5/n, exact=TRUE, x = matriz_entrenamiento, y = entrenamiento$psoda)[-1]
out_LASSO_5 <- fixedLassoInf(matriz_entrenamiento,entrenamiento$psoda, beta_LASSO_5 ,mejor_lambda_LASSO_5/n)
out_LASSO_5

# Eliminar todas las variables menos la 17,18,20

grid=10^seq(4,-2, length =100)
cv.mod=cv.glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=1,lambda=grid, nfolds = 5)
plot(cv.mod)
mejorlambda_4=cv.mod$lambda.min
mejorlambda_4

mod=glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda,alpha=1,lambda=grid)
pred=predict(mod,s=mejorlambda_4 ,newx=matriz_test_nueva)
error_4 <- sqrt(mean((pred-test$psoda )^2))
error_4

# LASSO with Elastic Net con cv10

beta_LASSO_EN_10 = coef(mejor_modelo, s=best_tune_EN_10$lambda/n, exact=TRUE, x = matriz_entrenamiento, y = entrenamiento$psoda)[-1]
out_LASSO_EN_10 <- fixedLassoInf(matriz_entrenamiento,entrenamiento$psoda, beta_LASSO_EN_10 ,best_tune_EN_10$lambda/n)
out_LASSO_EN_10

# Eliminar todas las variables menos la 17,18,20 que son la 18,19 y 21 en el data set entrenamiento

entrenamiento_nuevo <- entrenamiento[,c(1,18,19,21)]

Control <- trainControl(method = "cv", number = 10)
buscar_grid <- expand.grid(alpha = alpha_grid, lambda = lambda_grid)
entrenamiento_modelo <- train(psoda~., data = entrenamiento_nuevo, method = "glmnet", 
                              tuneGrid = buscar_grid, trControl = Control,
                              tuneLength = 10,
                              standardize = TRUE, maxit = 1000000)
entrenamiento_modelo$bestTune

modelo_glmnet <- entrenamiento_modelo$finalModel
coef(modelo_glmnet, s = entrenamiento_modelo$bestTune$lambda)
mejor_modelo <- glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda, alpha=entrenamiento_modelo$bestTune$alpha,
                       lambda = entrenamiento_modelo$bestTune$lambda, thresh = 1e-12)

pred_LASSO_elastic_1 <- predict(mejor_modelo,s=entrenamiento_modelo$bestTune$lambda,newx=matriz_test_nueva)
error_pred_LASSO_elastic_1 <- sqrt(mean((pred_LASSO_elastic_1 - test$psoda)^2))
error_pred_LASSO_elastic_1

# LASSO with Elastic Net con cv 5

beta_LASSO_EN_5 = coef(mejor_modelo, s=best_tune_EN_5$lambda/n, exact=TRUE, x = matriz_entrenamiento, y = entrenamiento$psoda)[-1]
out_LASSO_EN_5 <- fixedLassoInf(matriz_entrenamiento,entrenamiento$psoda, beta_LASSO_EN_5 ,best_tune_EN_5$lambda/n)
out_LASSO_EN_5

# Eliminar todas las variables menos la 17,18,20

Control <- trainControl(method = "cv", number = 5)
buscar_grid <- expand.grid(alpha = alpha_grid, lambda = lambda_grid)
entrenamiento_modelo <- train(psoda~., data = entrenamiento_nuevo, method = "glmnet", 
                              tuneGrid = buscar_grid, trControl = Control,
                              tuneLength = 10,
                              standardize = TRUE, maxit = 1000000)

entrenamiento_modelo$bestTune

modelo_glmnet <- entrenamiento_modelo$finalModel
coef(modelo_glmnet, s = entrenamiento_modelo$bestTune$lambda)
mejor_modelo <- glmnet(matriz_entrenamiento_nueva,entrenamiento$psoda, alpha=entrenamiento_modelo$bestTune$alpha,
                       lambda = entrenamiento_modelo$bestTune$lambda, thresh = 1e-12)

pred_LASSO_elastic_2 <- predict(mejor_modelo,s=entrenamiento_modelo$bestTune$lambda,newx=matriz_test_nueva)
error_pred_LASSO_elastic_2 <- sqrt(mean((pred_LASSO_elastic_2 - test$psoda)^2))
error_pred_LASSO_elastic_2

##### o)

# Penalización independiente
post_lasso_reg_indep = rlasso(entrenamiento$psoda~matriz_entrenamiento,post=TRUE, X.dependent.lambda = FALSE) 
print(post_lasso_reg_indep, all=FALSE)
yhat_postlasso_new_indep = predict(post_lasso_reg_indep, newdata=matriz_test)
error_postlasso_indep <- sqrt(mean((yhat_postlasso_new_indep - test$psoda )^2))
error_postlasso_indep

# Penalización dependiente
post_lasso_reg_dep = rlasso(entrenamiento$psoda~matriz_entrenamiento,post=TRUE, X.dependent.lambda = TRUE) 
print(post_lasso_reg_dep, all=FALSE)
yhat_postlasso_new_dep = predict(post_lasso_reg_dep, newdata=matriz_test)
error_postlasso_dep <- sqrt(mean((yhat_postlasso_new_dep - test$psoda )^2))
error_postlasso_dep

##### p) 

# Ambos modelos nos dan las mismas variables y los mismos coeficientes por lo que es igual para ambos.
lasso.effect = rlassoEffects(x=matriz_entrenamiento, y=entrenamiento$psoda, index=c("pfries", "NJ"), post = TRUE, )
print(lasso.effect)
summary(lasso.effect)

confint(lasso.effect, level=0.95, joint=TRUE)


plot(lasso.effect, main="Confidence Intervals")

##### q) escrito en el pdf

##### r)

pls_cv_5$coefficient


