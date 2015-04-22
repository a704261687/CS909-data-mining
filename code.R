
## read the file 
## change the one in your computer

Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jdk1.6.0_32\\jre')
dat <- read.csv("E:/360cloud/intime/project/20150412/reutersCSV.csv",
                stringsAsFactors = FALSE)

#################################################################
### -> CROSSVALIDATION ON TRAINING CORPUS
### Aim is to do xval w diff classifiers to find best performing
### classifiers and also feature selection methods.
### After this we can apply the best to testing data and cluster.
require(tm)
require(modeltools)
require(topicmodels)
require(e1071)
require(caret)
require(FSelector)
require(randomForest)
require(RTextTools)
require(slam)
require(proxy)
require(cluster)
require(fpc)


#Function that returns statistics (recall, accuracy etc..)
stats = function(conf){
  l = nrow(conf)
  #In case the matrix is not symmetric, add zeros appropriately
  if(nrow(conf)>ncol(conf)){
    for(i in 1:nrow(conf)){
      while(colnames(conf)[i]!=rownames(conf)[i]){
        if (i==1){
          conf = cbind(matrix(0, nrow=nrow(conf), ncol=1),conf[,i:ncol(conf)])
          
        } else {
          conf = cbind(conf[,1:(i-1)],matrix(0, nrow=nrow(conf), ncol=1),conf[,i:ncol(conf)])
        }
        colnames(conf)[1:i] = rownames(conf)[1:i]
        i=i+1
      }
    }
  }
  
  if(nrow(conf)<ncol(conf)){
    for(i in 1:ncol(conf)){
      while(colnames(conf)[i]!=rownames(conf)[i]){
        if (i==1){
          conf = cbind(matrix(0, ncol=ncol(conf), nrow=1),conf[i:nrow(conf),])
          
        } else {
          conf = cbind(conf[1:(i-1),],matrix(0, ncol=ncol(conf), nrow=1),conf[i:ncol(conf),])
        }
        rownames(conf)[i] = colnames(conf)[i]
        i=i+1
      }
    }
  }
  
  acc <- sum(diag(conf)) / sum(conf)
  precision.i <- diag(conf) / rowSums(conf)
  precision.i[is.na(precision.i)] <- 0
  recall.i <- diag(conf) / colSums(conf)
  recall.i[is.na(recall.i)] <- 0
  macro.p <- mean(precision.i)
  macro.r <- mean(recall.i)
  micro.p <- sum(precision.i * colSums(conf) / sum(conf))
  micro.r <- sum(recall.i * colSums(conf) / sum(conf))
  
  results = list()
  results$Accuracy = acc
  results$Macro.Precision = macro.p
  results$Macro.Recall = macro.r
  results$Micro.Precision = micro.p
  results$Micro.Recall = micro.r
  results$Classes = data.frame(Recall=recall.i,Precision=precision.i)
  rownames(results$Classes) = colnames(conf)
  results
}


# 1. Explore  the  data	and	undertake	any	cleaning/pre-processing	that	you	deem	necessary	
# for the	data	to	be	analysed
# names(dat)
# topicVar <- names(dat)[grepl("^topic.", names(dat))]
# topicDat <- subset(dat, select=topicVar)
# selRow <- apply(topicDat, 1, sum)
# selRow <- which(selRow == 0)
# dat <- dat[-selRow, ]
# topicDat <- topicDat[-selRow, ]
# 
# topicLDA <- LDA(topicDat, k=50)

# Build class vector for training/testing data for 10 most populous classes
classNames=paste0("topic.", c("earn","acq","money.fx","grain","crude",
                              "trade","interest","ship","wheat","corn"))

subDat <- subset(dat, select=c("purpose", "doc.text", classNames),
                 (purpose %in% c("train", "test")) &
                   (dat$doc.text != ""))

selectVar <- subset(subDat, select=c(classNames))
selectVar <- apply(selectVar, 1, sum)
subDat <- subDat[selectVar > 0, ]

vecDat <- vector()
vecClass <- vector()
vecPurpose <- vector()
for(i in 1:nrow(subDat)){
  tmp <- as.vector(subDat[i, classNames])
  tmpName1 <- names(tmp)[tmp > 0]
  
  tmpNum <- length(tmpName1)
  vecClass <- c(vecClass, tmpName1)
  vecPurpose <- c(vecPurpose, rep(subDat[i, "purpose"], tmpNum))
  vecDat <- c(vecDat, rep(subDat[i, "doc.text"], tmpNum))
}

vecDat <- VCorpus(VectorSource(vecDat))
vecClass <- gsub("topic.", "", vecClass)
# table(vecClass)
# table(gsub("topic.", "", vecClass))

re = vecDat
re = tm_map(re,tolower)
re = tm_map(re,removePunctuation)
re = tm_map(re,removeNumbers)
re = tm_map(re,removeWords,stopwords("english"))
re = tm_map(re,stripWhitespace)
re = tm_map(re,stemDocument)

for(i in 1:length(vecDat)){
  tmp <- re[[i]]
  re[[i]] <- vecDat[[i]]
  content(re[[i]]) <- tmp  
}




#Separate training/test sets by indexing
#Also Decide on one Topic for each Document and make a vector of topics corresponding to each document in the corpus
i.train = which(vecPurpose == "train")
i.test = which(vecPurpose == "test")
topics.list.train = vecClass[which(vecPurpose == "train")]
topics.list.test = vecClass[which(vecPurpose == "test")]

i.all = c(i.train,i.test)
topics.list = c(topics.list.train,topics.list.test)

n.train = length(i.train)
n.test = length(i.test)
n.all = length(i.all)

# topics.list.numeric = vector(length = n.all)
# for(i in 1:n.all){
#   topics.list.numeric[i] = make.numeric(topics.list[i])
# }
# 
# table(topics.list.numeric)



#FEATURE SELECTION
tf = DocumentTermMatrix(re)
tf.all = tf[i.all,];
tfidf = DocumentTermMatrix(re,control = c(weighting=weightTfIdf))
tfidf.all = tfidf[i.all,]

#Make graph for sparseness
sparseness = seq(0.6,0.99,by=0.01)
n.words = vector(length=length(sparseness))
count = 1;
for (i in sparseness){
  tf.test = removeSparseTerms(tf.all, i) 
  n.words[count] = dim(tf.test)[2];
  count = count + 1
}
plot(sparseness,n.words, main="sparseness", pch=20, cex=0.5)

#Decide on 0.9 (about 60 variables)
tf.all.s = removeSparseTerms(tf.all,0.9)
tfidf.all.s = removeSparseTerms(tfidf.all,0.9)

#Use tf, tfidf and LDA in conjuction
summary(col_sums(tfidf.all))
#Remove tokens from tf that have a tf-idf sum less than 0.1
tf.less = tf.all[,col_sums(tfidf.all)>0.1]

#Find important tokens using LDA
tf.lda = list(VEM = LDA(tf.less, k = 10, control = list(seed = 123)))
lda.terms <- unique(as.vector(terms(tf.lda[["VEM"]], 10)))
tfidf.less.lda = tfidf.all[,colnames(tfidf.all)%in%lda.terms]

#Get training and test sets for tf, tfidf and LDA(tfidf)
tf.train = as.data.frame(as.matrix(tf.all.s[1:n.train,]))
tf.train$class = topics.list.train
tf.test = as.data.frame(as.matrix(tf.all.s[(n.train + 1):(n.train+n.test),]))
tf.test$class = topics.list.test

tfidf.train = as.data.frame(as.matrix(tfidf.all.s[1:n.train,]))
tfidf.train$class = topics.list.train
tfidf.test = as.data.frame(as.matrix(tfidf.all.s[(n.train + 1):(n.train+n.test),]))
tfidf.test$class = topics.list.test

lda.train = as.data.frame(as.matrix(tfidf.less.lda[1:n.train,]))
lda.train$class = topics.list.train
lda.test = as.data.frame(as.matrix(tfidf.less.lda[(n.train + 1):(n.train+n.test),]))
lda.test$class = topics.list.test


#Compare Naive Bayes with Random Forests (using tf)
#Naive Bayes
# tf
nb.tf.model = naiveBayes(as.factor(class) ~ .,data = tf.train)
nb.tf.train = predict(nb.tf.model, tf.train)
nb.tf.train.tab = table(nb.tf.train,tf.train$class)
nb.tf.train.stat = stats(nb.tf.train.tab)

nb.tf.test = predict(nb.tf.model, tf.test)
nb.tf.test.tab = table(nb.tf.test,tf.test$class)
nb.tf.test.stat = stats(nb.tf.test.tab)


# tfidf
nb.tfidf.model = naiveBayes(as.factor(class) ~ .,data = tfidf.train)
nb.tfidf.train = predict(nb.tfidf.model, tfidf.train)
nb.tfidf.train.tab = table(nb.tfidf.train,tfidf.train$class)
nb.tfidf.train.stat = stats(nb.tfidf.train.tab)

nb.tfidf.test = predict(nb.tfidf.model, tfidf.test)
nb.tfidf.test.tab = table(nb.tfidf.test,tfidf.test$class)
nb.tfidf.test.stat = stats(nb.tfidf.test.tab)


# lda
nb.lda.model = naiveBayes(as.factor(class) ~ .,data = lda.train)
nb.lda.train = predict(nb.lda.model, lda.train)
nb.lda.train.tab = table(nb.lda.train,lda.train$class)
nb.lda.train.stat = stats(nb.lda.train.tab)

nb.lda.test = predict(nb.lda.model, lda.test)
nb.lda.test.tab = table(nb.lda.test,lda.test$class)
nb.lda.test.stat = stats(nb.lda.test.tab)

nb.sum.rest <- data.frame(TYPE=rep(c("Accuracy", "Precision", "Recall"), each=3),
                          feature=rep(c("TF", "TF-IDF", "LDA"), 3),
                          train=c(nb.tf.train.stat$Accuracy,nb.tfidf.train.stat$Accuracy,nb.lda.train.stat$Accuracy,
                                  nb.tf.train.stat$Macro.Precision,nb.tfidf.train.stat$Macro.Precision,nb.lda.train.stat$Macro.Precision,
                                  nb.tf.train.stat$Macro.Recall,nb.tfidf.train.stat$Macro.Recall,nb.lda.train.stat$Macro.Recall),
                          test=c(nb.tf.test.stat$Accuracy,nb.tfidf.test.stat$Accuracy,nb.lda.test.stat$Accuracy,
                                 nb.tf.test.stat$Macro.Precision,nb.tfidf.test.stat$Macro.Precision,nb.lda.test.stat$Macro.Precision,
                                 nb.tf.test.stat$Macro.Recall,nb.tfidf.test.stat$Macro.Recall,nb.lda.test.stat$Macro.Recall))

# SVM
# tf
svm.tf.model = svm(as.factor(class) ~ .,data = tf.train)
svm.tf.train = predict(svm.tf.model, tf.train)
svm.tf.train.tab = table(svm.tf.train,tf.train$class)
svm.tf.train.stat = stats(svm.tf.train.tab)

svm.tf.test = predict(svm.tf.model, tf.test)
svm.tf.test.tab = table(svm.tf.test,tf.test$class)
svm.tf.test.stat = stats(svm.tf.test.tab)


# tfidf
svm.tfidf.model = svm(as.factor(class) ~ .,data = tfidf.train)
svm.tfidf.train = predict(svm.tfidf.model, tfidf.train)
svm.tfidf.train.tab = table(svm.tfidf.train,tfidf.train$class)
svm.tfidf.train.stat = stats(svm.tfidf.train.tab)

svm.tfidf.test = predict(svm.tfidf.model, tfidf.test)
svm.tfidf.test.tab = table(svm.tfidf.test,tfidf.test$class)
svm.tfidf.test.stat = stats(svm.tfidf.test.tab)


# lda
svm.lda.model = svm(as.factor(class) ~ .,data = lda.train)
svm.lda.train = predict(svm.lda.model, lda.train)
svm.lda.train.tab = table(svm.lda.train,lda.train$class)
svm.lda.train.stat = stats(svm.lda.train.tab)

svm.lda.test = predict(svm.lda.model, lda.test)
svm.lda.test.tab = table(svm.lda.test,lda.test$class)
svm.lda.test.stat = stats(svm.lda.test.tab)

svm.sum.rest <- data.frame(TYPE=rep(c("Accuracy", "Precision", "Recall"), each=3),
                           feature=rep(c("TF", "TF-IDF", "LDA"), 3),
                           train=c(svm.tf.train.stat$Accuracy,svm.tfidf.train.stat$Accuracy,svm.lda.train.stat$Accuracy,
                                   svm.tf.train.stat$Macro.Precision,svm.tfidf.train.stat$Macro.Precision,svm.lda.train.stat$Macro.Precision,
                                   svm.tf.train.stat$Macro.Recall,svm.tfidf.train.stat$Macro.Recall,svm.lda.train.stat$Macro.Recall),
                           test=c(svm.tf.test.stat$Accuracy,svm.tfidf.test.stat$Accuracy,svm.lda.test.stat$Accuracy,
                                  svm.tf.test.stat$Macro.Precision,svm.tfidf.test.stat$Macro.Precision,svm.lda.test.stat$Macro.Precision,
                                  svm.tf.test.stat$Macro.Recall,svm.tfidf.test.stat$Macro.Recall,svm.lda.test.stat$Macro.Recall))


# randomforest
# tf
rf.tf.model = randomForest(as.factor(class) ~ .,data = tf.train)
rf.tf.train = predict(rf.tf.model, tf.train)
rf.tf.train.tab = table(rf.tf.train,tf.train$class)
rf.tf.train.stat = stats(rf.tf.train.tab)

rf.tf.test = predict(rf.tf.model, tf.test)
rf.tf.test.tab = table(rf.tf.test,tf.test$class)
rf.tf.test.stat = stats(rf.tf.test.tab)


# tfidf
rf.tfidf.model = randomForest(as.factor(class) ~ .,data = tfidf.train)
rf.tfidf.train = predict(rf.tfidf.model, tfidf.train)
rf.tfidf.train.tab = table(rf.tfidf.train,tfidf.train$class)
rf.tfidf.train.stat = stats(rf.tfidf.train.tab)

rf.tfidf.test = predict(rf.tfidf.model, tfidf.test)
rf.tfidf.test.tab = table(rf.tfidf.test,tfidf.test$class)
rf.tfidf.test.stat = stats(rf.tfidf.test.tab)


# lda
rf.lda.model = randomForest(as.factor(class) ~ .,data = lda.train)
rf.lda.train = predict(rf.lda.model, lda.train)
rf.lda.train.tab = table(rf.lda.train,lda.train$class)
rf.lda.train.stat = stats(rf.lda.train.tab)

rf.lda.test = predict(rf.lda.model, lda.test)
rf.lda.test.tab = table(rf.lda.test,lda.test$class)
rf.lda.test.stat = stats(rf.lda.test.tab)

rf.sum.rest <- data.frame(TYPE=rep(c("Accuracy", "Precision", "Recall"), each=3),
                           feature=rep(c("TF", "TF-IDF", "LDA"), 3),
                           train=c(rf.tf.train.stat$Accuracy,rf.tfidf.train.stat$Accuracy,rf.lda.train.stat$Accuracy,
                                   rf.tf.train.stat$Macro.Precision,rf.tfidf.train.stat$Macro.Precision,rf.lda.train.stat$Macro.Precision,
                                   rf.tf.train.stat$Macro.Recall,rf.tfidf.train.stat$Macro.Recall,rf.lda.train.stat$Macro.Recall),
                           test=c(rf.tf.test.stat$Accuracy,rf.tfidf.test.stat$Accuracy,rf.lda.test.stat$Accuracy,
                                  rf.tf.test.stat$Macro.Precision,rf.tfidf.test.stat$Macro.Precision,rf.lda.test.stat$Macro.Precision,
                                  rf.tf.test.stat$Macro.Recall,rf.tfidf.test.stat$Macro.Recall,rf.lda.test.stat$Macro.Recall))








# # SVM
# svm.model <- svm(as.factor(class) ~ .,data=tf.train)
# svm.tf.train = predict(svm.model, tf.train)
# svm.tf.conf = table(svm.tf.train,tf.train$class)
# stats(svm.tf.conf)
# 
# #RANDOMFOREST
# rf.tf.pred <- randomForest(as.factor(class) ~ .,data=tf.train, importance=TRUE,proximity=TRUE)
# rf.tf.train = predict(rf.tf.pred,tf.train)
# rf.tf.conf = table(rf.tf.train,tf.train$class)
# stats(rf.tf.conf)
# 
# 
# #Compare TF,TFIDF and LDA using Random Forests
# rf.tfidf.pred <- randomForest(as.factor(class) ~ .,data = tfidf.train, importance=TRUE,proximity=TRUE)
# rf.tfidf.train = predict(rf.tfidf.pred,tfidf.train)
# rf.tfidf.conf = table(rf.tfidf.train,tfidf.train$class)
# stats(rf.tfidf.conf)
# 
# rf.lda.pred <- randomForest(as.factor(class) ~ .,data = lda.train, importance=TRUE,proximity=TRUE)
# rf.lda.train = predict(rf.lda.pred,lda.train)
# rf.lda.conf = table(rf.lda.train,lda.train$class)
# stats(rf.lda.conf)

# #Run best one for test set
# rf.tfidf.test = predict(rf.tfidf.pred, tfidf.test)
# rf.tfidf.conf.test = table(rf.tfidf.test,tfidf.test$class)
# stats(rf.tfidf.conf.test)






#######################
## cross-validation
nFolds <- 10
numDocs <- nrow(lda.train)
reordered <- sample(numDocs)
folds <- cut(1:numDocs,breaks=nFolds,labels=F)

accuracy.nb <- NULL
accuracy.svm <- NULL
accuracy.rf <- NULL

precision.nb <- NULL
precision.svm <- NULL
precision.rf <- NULL

recall.nb <- NULL
recall.svm <- NULL
recall.rf <- NULL


for(i in 1:10){
  train.dat <- lda.train[reordered[folds!=i], ]
  test.dat <- lda.train[reordered[folds==i], ]
  
  #Naive Bayes
  nb.lda.model = naiveBayes(as.factor(class) ~ ., data = train.dat)
  nb.lda.test = predict(nb.lda.model, test.dat)
  nb.lda.tab = table(nb.lda.test, test.dat$class)
  nb.stat <- stats(nb.lda.tab)
  accuracy.nb <- c(accuracy.nb, nb.stat$Accuracy)
  precision.nb <- c(precision.nb, nb.stat$Macro.Precision)
  recall.nb <- c(recall.nb, nb.stat$Macro.Recall)
  
  # SVM
  svm.model <- svm(as.factor(class) ~ .,data=train.dat)
  svm.test = predict(svm.model, test.dat)
  svm.tab = table(svm.test, test.dat$class)
  svm.stat <- stats(svm.tab)
  accuracy.svm <- c(accuracy.svm, svm.stat$Accuracy)
  precision.svm <- c(precision.svm, svm.stat$Macro.Precision)
  recall.svm <- c(recall.svm, svm.stat$Macro.Recall)
  
  #RANDOMFOREST
  rf.model <- randomForest(as.factor(class) ~ .,data=train.dat, importance=TRUE,proximity=TRUE)
  rf.test = predict(rf.model, test.dat)
  rf.tab = table(rf.test, test.dat$class)
  rf.stat <- stats(rf.tab)
  accuracy.rf <- c(accuracy.rf, rf.stat$Accuracy)
  precision.rf <- c(precision.rf, rf.stat$Macro.Precision)
  recall.rf <- c(recall.rf, rf.stat$Macro.Recall)
}


plot(accuracy.nb, type="l", 
     ylim=c(0.3, 1), lty=2, col="red",
     xlab="N-fold", ylab="accuracy", main="ACCURACY")
lines(accuracy.svm, lty=3, col="blue")
lines(accuracy.rf, lty=4, col="black")
legend("top", lty=2:4, col=c("red","blue","black"),
       legend=c("nb", "svm", "rf"))


plot(precision.nb, type="l", 
     ylim=c(0.3, 1), lty=2, col="red",
     xlab="N-fold", ylab="precision", main=toupper("precision"))
lines(precision.svm, lty=3, col="blue")
lines(precision.rf, lty=4, col="black")
legend("top", lty=2:4, col=c("red","blue","black"),
       legend=c("nb", "svm", "rf"))

plot(recall.nb, type="l", 
     ylim=c(0.3, 1), lty=2, col="red",
     xlab="N-fold", ylab="recall", main=toupper("recall"))
lines(recall.svm, lty=3, col="blue")
lines(recall.rf, lty=4, col="black")
legend("top", lty=2:4, col=c("red","blue","black"),
       legend=c("nb", "svm", "rf"))


mean.accuracy.nb <- mean(accuracy.nb)
mean.accuracy.svm <- mean(accuracy.svm)
mean.accuracy.rf <- mean(accuracy.rf)

mean.precision.nb <- mean(precision.nb)
mean.precision.svm <- mean(precision.svm)
mean.precision.rf <- mean(precision.rf)

mean.recall.nb <- mean(recall.nb)
mean.recall.svm <- mean(recall.svm)
mean.recall.rf <- mean(recall.rf)


############
# from the above plot, SVM model may be the "optimal"  model, because of the accuracy,	precision	and	recall of SVM better than others
# the	"optimal"	model for training data
# SVM
svm.model <- svm(as.factor(class) ~ .,data=lda.train)
svm.train.pred = predict(svm.model, lda.train)
svm.test.pred = predict(svm.model, lda.test)
svm.train.tab = table(svm.train.pred, lda.train$class)
svm.test.tab = table(svm.test.pred, lda.test$class)

svm.train.stat = stats(svm.train.tab)
svm.test.stat = stats(svm.test.tab)

# estimate of  the	accuracy,	precision	and	recall of the test data
accuracy.test <- svm.test.stat$Accuracy
precision.test <- svm.test.stat$Macro.Precision
recall.test <- svm.test.stat$Macro.Recall

# micro and macro averaged precision	and	recall of the train data
mac.precision.train <- svm.train.stat$Macro.Precision
mic.precision.train <- svm.train.stat$Micro.Precision
mac.recall.train <- svm.train.stat$Macro.Recall
mic.recall.train <- svm.train.stat$Micro.Recall

### the difference micro and macro averaged
# Macroaveraging computes a simple average over classes. Microaveraging pools per-document decisions across classes,
# and then computes an effectiveness measure on the pooled contingency table


######################
# Now  consider	all	the	data	and	use	the	best	performing	features	of	part	(3)	to	represent
# the	documents and	apply	three	clustering	algorithms	of	your	choice, selected	from	
# those discussed	in	lectures.	Justify	your	choice.	Provide	appropriate	measures	of	
# cluster	quality.	Is	there	a	correspondence	between	clusters	and	the	original	TOPICS	
# labels?


# Compute feature relevance measure for each term
spisok <- chi.squared(class~., lda.train)

# (Sort? and) Select the most highly relevant features (in some defined way...)
#sorted <- spisok[order(spisok[,"attr_importance"]), , drop=FALSE]
bestPerfFeats <- cutoff.k(spisok, 40)

#CLUSTERING
clus.dat <- rbind(lda.test, lda.train)
clus.dat <- subset(clus.dat, select=bestPerfFeats)
clus.prcomp <- prcomp(clus.dat)

# Hierarchical Agglomerative 
distance<- dist(clus.dat, method = "euclidean") # or binary,canberra, maximum, manhattan
fit1 <- hclust(distance, method="ward")
groups <- cutree(fit1, k=10) # cut tree into 10 clusters
groups1 <- as.data.frame(groups)
plot(fit1, labels = NULL, hang = 0.1,
     axes = TRUE, frame.plot = FALSE, ann = TRUE,
     main = "Cluster Dendrogram",
     sub = NULL, xlab = NULL, ylab = "Height") # display dendogram
rect.hclust(fit1, k=10, border="red")

plot(clus.prcomp$x, col=groups, pch=20, cex=0.5, 
     main="Hierarchical")

allClasses <- c(lda.train$class, lda.test$class)
table1<- table(groups1$groups,allClasses)
kable(table1, row.names=T)


# K-means
wssplot <- function(data, nc=15, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}
#df <- scale(clus.dat)
wssplot(clus.dat) 

fit2 <- kmeans(clus.dat, 10)
plot(clus.prcomp$x, col=fit2$cl,pch=20, cex=0.5, 
     main="K-means")

groups2<-as.data.frame(fit2$cl)
colnames(groups2)<-"groups"
table2<- table(groups2$groups,allClasses)
kable(table2, row.names=T)



# EM clustering
library(mclust)
fit3 <- Mclust(clus.dat,G=10)
plot(clus.prcomp$x, col=fit3$cl,pch=20, cex=0.5, 
     main="E-M clustering")

groups3<-as.data.frame(fit3$cl)
colnames(groups3)<-"groups"
table3<- table(groups3$groups,allClasses)
kable(table3, row.names=T)

# display the best model
summary(fit3) 
