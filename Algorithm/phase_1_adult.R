# import libraries
library(lpSolve)
library(tidyverse)
library(pROC)

# import Adult dataset
# The data is available at https://archive.ics.uci.edu/ml/datasets/adult
raw.train <- read.table("adultdata.txt", header = F, sep = ",")
raw.test <- read.table("adulttest.txt", header = F, sep = ",")
raw.train <- raw.train[, c(1, 5, 9, 10, 15)]
raw.test <- raw.test[, c(1, 5, 9, 10, 15)]
raw.df <- rbind(raw.train, raw.test)
names(raw.df) <- c("age", "education", "race", "gender", "income")
nn <- nrow(raw.df)

# feature transformation
# age
age <- raw.df$age
age <- (age %/% 10) * 10
age[which(age >= 70)] = ">=70"

# education
education <- raw.df$education
smaller.than.6 <- which(education <= 5)
bigger.than.12 <- which(education >= 13)
education[smaller.than.6] = "<6"
education[bigger.than.12] = ">12"
education <- factor(education, levels = c("<6", "6", "7", "8", "9", "10", "11", "12", ">12"))

# race
race <- rep(NA, nn)
white <- which(raw.df$race == levels(raw.df$race)[5])
non.white <- which(raw.df$race != levels(raw.df$race)[5])
race[white] = "White"
race[non.white] = "Non-white"
race <- as.factor(race)

# gender
gender <- rep(NA, nn)
female <- which(raw.df$gender == levels(raw.df$gender)[1])
male <- which(raw.df$gender == levels(raw.df$gender)[2])
gender[female] = "Female"
gender[male] = "Male"
gender <- as.factor(gender)

# income
income <- rep(NA, nn)
leq.than.50K <- which(raw.df$income == levels(raw.df$income)[1] | 
                        raw.df$income == levels(raw.df$income)[3])
greater.than.50K <- which(raw.df$income == levels(raw.df$income)[2] |
                            raw.df$income == levels(raw.df$income)[4])
income[leq.than.50K] = "<=50K"
income[greater.than.50K] = ">50K"
income <- as.factor(income)

df <- data.frame(age, education, race, gender, income)
names(df) <- c("age", "education", "race", "gender", "income")

# 5-fold cross validation
trun <- rep(list(list()), 5)
all <- 1:nn
for (i in 1:4) {
  trun[[i]] <- (1+(i-1)*(nn %/% 5)):(i*(nn %/% 5))
}
trun[[5]] <- (1+4*(nn %/% 5)):nn

# RRW implementation
RRW <- function(lambda) {
  
  output <- matrix(0, ncol = 2, nrow = 5)
  rownames(output) <- 1:5
  colnames(output) <- c("auc", "j")
  
  output2 <- matrix(0, ncol = 2, nrow = 3)
  rownames(output2) <- c("mean-se", "mean", "mean+se")
  colnames(output2) <- c("AUC Full", "AUC Reweigh", "AUC Old", "J Full", "J Reweigh", "J Old")
  
  for (i in 1:5) {
    
    testdf <- df[trun[[i]], ]
    traindf <- df[all[-trun[[i]]], ]
    
    compare <- traindf %>%
      dplyr::group_by(age, education, race, gender, income, .drop = FALSE) %>%
      summarise(n = n())
    
    n_f = sum(traindf$gender == "Female")
    n_m = sum(traindf$gender == "Male")
    n = nrow(traindf)
    n_0 = sum(traindf$income == "<=50K")
    n_1 = sum(traindf$income == ">50K")
    N = 504
    N_x = 126
    N_y = 2
    N_d = 2
    
    compare2 <- traindf %>% count(gender, income, sort = FALSE)
    compare2$wts <- c(n_0*n_f/n/sum(traindf$gender == "Female" & traindf$income == "<=50K"), 
                      n_1*n_f/n/sum(traindf$gender == "Female" & traindf$income == ">50K"),
                      n_0*n_m/n/sum(traindf$gender == "Male" & traindf$income == "<=50K"),
                      n_1*n_m/n/sum(traindf$gender == "Male" & traindf$income == ">50K"))
    
    f.obj <- c(rep(0, N), rep(1, 1.5*N))
    
    compare.xd <- traindf %>%
      dplyr::group_by(age, education, race, gender, .drop = FALSE) %>%
      summarise(n = n())
    compare.xd$n[which(compare.xd$n == 0)] <- 1
    f.con_ul <- kronecker(diag(N_x*N_d), matrix(c(1, 1, -1, -1), nrow = 2)) %*% diag(compare$n/rep(compare.xd$n, each = 2))
    f.con_um <- kronecker(diag(N_x*N_d), matrix(c(-1, 1), ncol = 1))
    f.con_ur <- matrix(0, ncol = N, nrow = N)
    
    f.con_ml <- kronecker(diag(N), matrix(rep(lambda, 2), ncol = 1))
    f.con_mm <- matrix(0, ncol = N_x*N_y, nrow = 2*N)
    f.con_mr <- kronecker(diag(N), matrix(c(-1, 1), ncol = 1))
    
    f.con_ll1 <- kronecker(matrix(1, ncol = N_x), diag(N_y*N_d)) %*% diag(compare$n)
    f.con_ll2 <- kronecker(matrix(1, ncol = N_x), kronecker(diag(N_y), matrix(1, ncol = N_d))) %*% diag(compare$n)
    f.con_ll3 <- kronecker(matrix(1, ncol = N_x), kronecker(matrix(1, ncol = N_d), diag(N_y))) %*% diag(compare$n)
    f.con_lr <- matrix(0, ncol = N_x*N_y+N, nrow = N_y*N_d+N_y+N_d)
    
    f.dir <- c(rep(c("<=", ">="), 1.5*N), rep("=", N_y*N_d+N_y+N_d))
    
    f.con <- rbind(cbind(f.con_ul, f.con_um, f.con_ur),
                   cbind(f.con_ml, f.con_mm, f.con_mr),
                   cbind(rbind(f.con_ll1, f.con_ll2, f.con_ll3),
                         f.con_lr))
    
    f.rhs <- c(rep(0, N), 
               rep(rep(compare2$wts, N_x), each = N_d)*lambda, 
               n_f*n_0/n, n_f*n_1/n, n_m*n_0/n, n_m*n_1/n, n_f, n_m, n_0, n_1)
    
    compare$wts <- lp("min", f.obj, f.con, f.dir, f.rhs)$solution[1:N]
    
    traindf.transformed <- merge(x=traindf, y=compare, 
                                 by = c("age", "education", "race", "income", "gender"), 
                                 all.x = T, sort = F)
    
    LR.reweigh <- glm(income ~ age + education + race + gender, 
                      family = "binomial", 
                      data = traindf.transformed,
                      weight = wts)
    
    LR.reweigh.pred <- predict.glm(LR.reweigh, testdf, type = "response")
    
    f.test.index <- which(testdf$gender == "Female")
    m.test.index <- which(testdf$gender == "Male")
    p.f.reweigh <- mean(LR.reweigh.pred[f.test.index])
    p.m.reweigh <- mean(LR.reweigh.pred[m.test.index])
    
    output[i, 1] <- auc(testdf$income, LR.reweigh.pred)
    output[i, 2] <- max(abs(p.f.reweigh/p.m.reweigh - 1), abs(p.m.reweigh/p.f.reweigh - 1))
    
  }
  
  se <- apply(output, 2, sd)/sqrt(5)
  output2[2, ] <- apply(output, 2, mean)
  output2[1, ] <- output2[2, ] - se
  output2[3, ] <- output2[2, ] + se
  
  outcome <- cbind(lambda, matrix(output2, ncol = 6, byrow = FALSE))
  colnames(outcome) <- c("lambda", "auc.lower", "auc", "auc.upper", "j.lower", "j", "j.upper")
  outcome
  
}

# experiment
RRW(0.7)