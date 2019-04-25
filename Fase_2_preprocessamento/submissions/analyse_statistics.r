

#teste de shapiro wilk
all_p <- shapiro.test(all_filters_submission$Saleprice)$p.value
corr_p <- shapiro.test(corr_filter_submission$Saleprice)$p.value
kbest_p <- shapiro.test(KBest_wrapper_submission$Saleprice)$p.value
nan_p <- shapiro.test(KBest_wrapper_submission$Saleprice)$p.value
rfe_p <- shapiro.test(rfe_wrapper$Saleprice)$p.value
unb_p<- shapiro.test(unbalanced_filter$Saleprice)$p.value

shapiro_df <- data.frame('al'=all_p, 'corr'=corr_p, 'kbest'=kbest_p, 'nan'=nan_p, 'rfe'=rfe_p, 'unb'=unb_p)

value <- c(all_filters_submission$Saleprice,corr_filter_submission$Saleprice,KBest_wrapper_submission$Saleprice,KBest_wrapper_submission$Saleprice,rfe_wrapper$Saleprice,unbalanced_filter$Saleprice)

n <- 6
k <- length(value)/n
len <- length(value)

z <- gl(n,k,len,labels = c('al','corr','kbest','nan','rfe','unb'))

m <- matrix(value,
            nrow = k,
            ncol=n,
            byrow = TRUE,
            dimnames = list(1 : k,c('al','corr','kbest','nan','rfe','unb')))


f <- friedman.test(m) 

fp <- posthoc.friedman.nemenyi.test(m)

nt <- NemenyiTest(value,z, out.list = TRUE)