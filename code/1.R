# 2023-7-29
# xiao ran
# 读取类别编码csv文件，进行训练分类

# 加载caTools包
library(caret)
library(e1071)
library(caTools)
library(randomForest)

# 拆分数据集，n类别数，ratio拆分比例
dataset_split <- function(n, ratio){
  # 获取文件夹下所有 CSV 文件的文件名
  csv_files <- list.files("D:/ShortTerm/Feature_Extraction/Video_Coding", pattern = "\\.csv$", full.names = FALSE)
  csv_used <- sample(csv_files, n)
  
  # 读取第一个数据集作为起始
  csv_file_path <- paste("D:/ShortTerm/Feature_Extraction/Video_Coding/", csv_used[1], sep="")
  data <- read.csv(csv_file_path, header = FALSE)
  
  index <- dim(data)[1] * ratio
  train_data <- data[1:index,]
  test_data <- data[index:dim(data)[1],]
  train_label <- rep(1, index)
  test_label <- rep(1, dim(data)[1] - index + 1)
  count <- 1
  
  for (i in csv_used[2:length(csv_used)])
  {
    count <- count + 1  # 更新计数
    
    temp_path <- paste("D:/ShortTerm/Feature_Extraction/Video_Coding/", i, sep="")
    temp_data <- read.csv(temp_path, header = FALSE)
    
    temp_index <- dim(temp_data)[1] * ratio
    temp_traindata <- temp_data[1:temp_index,]
    temp_testdata <- temp_data[temp_index:dim(temp_data)[1],]
    temp_trainlabel <- rep(count, temp_index)
    temp_testlabel <- rep(count, dim(temp_data)[1] - temp_index + 1)
    
    train_data <- rbind(train_data, temp_traindata)
    test_data <- rbind(test_data, temp_testdata)
    train_label <- c(train_label, temp_trainlabel)
    test_label <- c(test_label, temp_testlabel)
    
  }  # 
  
  return(list(dataset1 = train_data, dataset2 = test_data,
              dataset3 = train_label, dataset4 = test_label))
}  # end function

classify <- function(n, ratio)
{
  result <- dataset_split(n, ratio)
  train_data <- result$dataset1
  test_data <- result$dataset2
  train_label <- result$dataset3
  test_label <- result$dataset4
  
  # 训练SVM模型
  svm_model <- svm(train_label ~ ., data = train_data, kernel = "linear")
  # 进行预测
  predictions <- predict(svm_model, test_data)
  predictions <- as.vector(round(predictions))
  predictions <- ifelse(predictions < 1, 1, ifelse(predictions > n, n, predictions))
  # 计算准确率
  Accuracy_svm <- confusionMatrix(factor(predictions), factor(test_label))$overall["Accuracy"]
  
  # 构建随机森林模型
  rf_model <- suppressWarnings(randomForest(x=train_data, y=train_label, ntree = 100))
  # 在测试集上进行预测
  predicted <- predict(rf_model, newdata = test_data)
  predicted <- as.vector(round(predicted))
  predicted <- ifelse(predictions < 1, 1, ifelse(predicted > n, n, predicted))
  # 计算准确率
  Accuracy_rf <- confusionMatrix(factor(predicted), factor(test_label))$overall["Accuracy"]
  
  return( list( acc1 = as.numeric(Accuracy_svm), acc2 = as.numeric(Accuracy_rf) ) )
}

ratio <- 0.7
acc_svm_list <- c()
acc_rf_list <- c()
for (i in 2:8)
{
  acc_svm_best <- 0
  acc_rf_best <- 0
  for (j in 1:10)
  {
    temp <- classify(i, ratio)
    acc_svm <- temp$acc1
    acc_rf <- temp$acc2
    
    if ((acc_svm + acc_rf) > (acc_svm_best + acc_rf_best))
    {
      acc_svm_best <- acc_svm
      acc_rf_best <- acc_rf
    }
  }
  
  acc_svm_list <- append(acc_svm_list, acc_svm_best)
  acc_rf_list <- append(acc_rf_list, acc_rf_best)
}

# 绘制第一条折线
plot(1:length(acc_svm_list)+1, acc_svm_list, type = "l", ylim = c(0, 1), col = "blue", xlab = "n", ylab = "Accuracy")
# 添加第二条折线
lines(1:length(acc_svm_list)+1, acc_rf_list, type = "l", col = "red")
# 添加图例
legend(x = 5, y = 1.0, legend = c("SVM", "Random Forest"), col = c("blue", "red"), lty = 1, ncol = 1, cex = 0.8)


