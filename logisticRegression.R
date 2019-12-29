install.packages("ggplot2")
library(ggplot2)
install.packages("caTools")
library(caTools)
install.packages("caret")
library(caret)

# Logistic Regression


# Import the dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset <- dataset[,3:5]


# Plotting to see the relation between points
ggplot(dataset, aes(dataset[,1], dataset[,3])) + geom_point() + ggtitle('Visualisation') + xlab('Age') + ylab('Purchased')
ggplot(dataset, aes(dataset[,2], dataset[,3])) + geom_point() + ggtitle('Visualisation') + xlab('Estimated Salary') + ylab('Purchased')


# Splitting the dataset into train and test sets
set.seed(123)
sample = sample.split(dataset, SplitRatio = 3/4)
train = subset(dataset, sample == TRUE)
test = subset(dataset, sample == FALSE)


# Standard Scaling in R
train[,1:2] = scale(train[,1:2])
test[,1:2] = scale(test[,1:2])


# Fitting Logistic Regression to our dataset
model = glm(formula = Purchased ~ Age + EstimatedSalary, family = binomial, data = train)


# Predicting the results
prediction = predict(model, newdata = test[,-3], type = "response")
predicted = ifelse(prediction > 0.5, 1, 0)


# Making the confusion matrix
table(test[,3], predicted)