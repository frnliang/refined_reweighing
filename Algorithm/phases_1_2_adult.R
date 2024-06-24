# import libraries
library(lpSolve)
library(tidyverse)
library(dplyr)
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

# age
age <- raw.df$age

# education
education <- raw.df$education

# feature transformation
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
  
  output1 <- matrix(0, ncol = 2, nrow = 5)
  output2 <- rep(0, 6)
  
  for (I in 1:5) {
    
    testdf <- df[trun[[I]], ]
    traindf <- df[all[-trun[[I]]], ]
    
    compare <- traindf %>%
      dplyr::group_by(income, race, gender, .drop = FALSE) %>%
      summarise(n = n())
    
    n_f = sum(traindf$gender == "Female")
    n_m = sum(traindf$gender == "Male")
    n = nrow(traindf)
    n_0 = sum(traindf$income == "<=50K")
    n_1 = sum(traindf$income == ">50K")
    N = 8
    N_x = 2
    N_y = 2
    N_d = 2
    
    compare2 <- traindf %>% count(gender, income, sort = FALSE)
    compare2$wts <- c(n_0*n_f/n/sum(traindf$gender == "Female" & traindf$income == "<=50K"), 
                      n_1*n_f/n/sum(traindf$gender == "Female" & traindf$income == ">50K"),
                      n_0*n_m/n/sum(traindf$gender == "Male" & traindf$income == "<=50K"),
                      n_1*n_m/n/sum(traindf$gender == "Male" & traindf$income == ">50K"))
    
    f.obj <- c(rep(0, N), rep(1, 1.5*N))
    
    compare.xd <- traindf %>%
      dplyr::group_by(race, gender, .drop = FALSE) %>%
      summarise(n = n())
    compare.x <- traindf %>%
      dplyr::group_by(race, .drop = FALSE) %>%
      summarise(n = n())
    compare.xd$n[which(compare.xd$n == 0)] <- 1
    compare.x$n[which(compare.x$n == 0)] <- 1
    f.con_ul.1 <- kronecker(diag(N_x*N_d), matrix(c(1, 1, -1, -1), nrow = 2))
    f.con_ul.2 <- diag(compare$n/rep(compare.xd$n, 2))
    f.con_ul.3 <- diag(rep(rep(compare.x$n, each = N_d), N_y)) / n
    f.con_ul <- f.con_ul.1 %*% f.con_ul.2 %*% f.con_ul.3
    
    compare <- traindf %>%
      dplyr::group_by(race, gender, income, .drop = FALSE) %>%
      summarise(n = n())
    
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
                                 by = c("race", "income", "gender"), 
                                 all.x = T, sort = F)
    
    cutoff.age <- rep(NA, N)
    cutoff.edu <- rep(NA, N)
    
    ratio.age <- matrix(NA, ncol = 18, nrow = N)
    ratio.edu <- matrix(NA, ncol = 16, nrow = N)
    
    count.age <- matrix(NA, ncol = 18, nrow = N)
    count.edu <- matrix(NA, ncol = 16, nrow = N)
    
    for (j in 1:N) {
      
      subdata <- traindf.transformed[which(traindf.transformed$wts == compare$wts[j]), ]
      
      # get the ratio of age
      
      age <- subdata$age - 17
      subdata$age.index <- age %/% 4 + 1
      subdata$age.index[which(subdata$age.index == 19)] <- 18
      
      freq.age <- rep(0, 18)
      for (a in 1:length(age)) {
        b <- subdata$age.index[a]
        freq.age[b] <- freq.age[b] + 1
      }
      count.age[j, ] <- freq.age
      
      # binary search
      
      max.dev <- max(freq.age)
      div <- max.dev%/%2
      lower <- min(freq.age)
      upper <- max.dev
      
      while (upper - lower > 1) {
        inter <- freq.age - div
        ratio <- sum(inter)
        if (ratio < 0) {
          upper <- div
          div <- (lower + div)%/%2
        } else if (ratio > 0) {
          lower <- div
          div <- (upper + div)%/%2
        } else {
          break
        }
      }
      
      division <- c(div-1, div, div+1)
      diff <- rep(NA, 3)
      for (k in 1:3) {
        inter <- freq.age - division[k]
        diff[k] <- abs(sum(inter))
      }
      cutoff.age[j] <- division[which.min(diff)]
      
      max.dev <- max(freq.age-cutoff.age[j])
      dev.rat <- (freq.age-cutoff.age[j])/max.dev
      
      ratio.age[j, ] <- dev.rat
      
      # get the ratio of education
      
      edu <- subdata$education
      freq.edu <- rep(0, 16)
      for (a in 1:length(edu)) {
        b <- edu[a]
        freq.edu[b] <- freq.edu[b] + 1
      }
      count.edu[j, ] <- freq.edu
      
      # binary search
      
      max.dev <- max(freq.edu)
      div <- max.dev%/%2
      lower <- min(freq.edu)
      upper <- max.dev
      
      while (upper - lower > 1) {
        inter <- freq.edu - div
        ratio <- sum(inter)
        if (ratio < 0) {
          upper <- div
          div <- (lower + div)%/%2
        } else if (ratio > 0) {
          lower <- div
          div <- (upper + div)%/%2
        } else {
          break
        }
      }
      
      division <- c(div-1, div, div+1)
      diff <- rep(NA, 3)
      for (k in 1:3) {
        inter <- freq.edu - division[k]
        diff[k] <- abs(sum(inter))
      }
      cutoff.edu[j] <- division[which.min(diff)]
      
      max.dev <- max(freq.edu-cutoff.edu[j])
      dev.rat <- (freq.edu-cutoff.edu[j])/max.dev
      
      ratio.edu[j, ] <- dev.rat
      
    }
    
    count.age.base <- matrix(NA, nrow = N/2, ncol = 18)
    for (i in 1:(N/2)) {
      count.age.base[i, ] <- colSums(count.age[(2*i-1):(2*i), ])
    }
    count.age.base <- kronecker(count.age.base, matrix(c(1, 1), ncol = 1))
    
    count.edu.base <- matrix(NA, nrow = N/2, ncol = 16)
    for (i in 1:(N/2)) {
      count.edu.base[i, ] <- colSums(count.edu[(2*i-1):(2*i), ])
    }
    count.edu.base <- kronecker(count.edu.base, matrix(c(1, 1), ncol = 1))
    
    cond.age <- count.age / count.age.base
    cond.age[is.na(cond.age)] <- 0
    cond.edu <- count.edu / count.edu.base
    cond.edu[is.na(cond.edu)] <- 0
    
    order <- matrix(c(1, 3, 2, 4, 5, 7, 6, 8), nrow = 4, byrow = T)
    
    cal <- function(tuning) {
      
      W.age <- matrix(rep(compare$wts, rep(18, N)), nrow = N, byrow = T) * (1-tuning*ratio.age)
      W.edu <- matrix(rep(compare$wts, rep(16, N)), nrow = N, byrow = T) * (1-tuning*ratio.edu)
      
      W.cond.age <- cond.age * W.age
      W.cond.edu <- cond.edu * W.edu
      
      objective <- 0
      for (i in 1:4) {
        a <- order[i, 1]
        b <- order[i, 2]
        int.age <- sum(abs(W.cond.age[a, ]-W.cond.age[b, ]))
        int.edu <- sum(abs(W.cond.edu[a, ]-W.cond.edu[b, ]))
        objective <- objective + int.age + int.edu
      }
      
      return(objective)
      
    }
    
    # binary search
    
    lower <- 0
    upper <- 1
    
    while (upper - lower > 0.000001) {
      cal.lower <- cal(lower)
      cal.upper <- cal(upper)
      if (cal.lower < cal.upper) {
        upper <- (lower + upper) / 2
      } else {
        lower <- (lower + upper) / 2
      } 
    }
    opt <- lower
    
    W.age.opt <- matrix(rep(compare$wts, rep(18, N)), nrow = N, byrow = T) * (1-opt*ratio.age)
    W.edu.opt <- matrix(rep(compare$wts, rep(16, N)), nrow = N, byrow = T) * (1-opt*ratio.edu)
    
    W.opt <- list()
    z <- 1
    while (z < 9) {
      W.opt.sub <- matrix(NA, nrow = 18, ncol = 16)
      for (i in 1:18) {
        for (j in 1:16) {
          W.opt.sub[i, j] <- (W.age.opt[z, i] + W.edu.opt[z, j]) / 2
        }
      }
      W.opt[[z]] <- W.opt.sub
      z <- z + 1
    }
    
    age.index <- 1:18
    edu.index <- 1:16
    compare.full <- merge(edu.index, compare[, 1:3], by = NULL)
    compare.full <- merge(age.index, compare.full, by = NULL)
    names(compare.full)[1:2] <- c("age.index", "education")
    
    W.flat <- c()
    for (z in 1:8) {
      W.flat <- c(W.flat, as.vector(W.opt[[z]]))
    }
    
    compare.full$wts <- W.flat
    
    newage <- traindf$age - 17
    traindf$age.index <- newage %/% 4 + 1
    traindf$age.index[which(traindf$age.index == 19)] <- 18
    
    traindf.new <- merge(x=traindf, y=compare.full, 
                         by = c("education", "race", "gender", "income", "age.index"), 
                         all.x = T, sort = F)
    
    # new reweighing
    LR.reweigh <- glm(income ~ age + education + race + gender, 
                      family = "binomial", 
                      data = traindf.new,
                      weight = wts)
    
    LR.reweigh.pred <- predict.glm(LR.reweigh, testdf, type = "response")
    
    output1[I, 1] <- auc(testdf$income, LR.reweigh.pred)
    
    f.test.index <- which(testdf$gender == "Female")
    m.test.index <- which(testdf$gender == "Male")
    p.f.reweigh <- mean(LR.reweigh.pred[f.test.index])
    p.m.reweigh <- mean(LR.reweigh.pred[m.test.index])
    
    output1[I, 2] <- max(abs(p.f.reweigh/p.m.reweigh - 1), abs(p.m.reweigh/p.f.reweigh - 1))
    
  }
  
  se <- apply(output1, 2, sd)/sqrt(5)
  output2[c(2, 5)] <- apply(output1, 2, mean)
  output2[1] <- output2[2] - se[1]
  output2[3] <- output2[2] + se[1]
  output2[4] <- output2[5] - se[2]
  output2[6] <- output2[5] + se[2]
  
  outcome <- c(lambda, output2)
  outcome
  
}

# experiment
RRW(0.7)