library(reshape2)
library(ggplot2)


Lipify <- read.csv("Project_Accuracy.csv", header = TRUE)


# Test Accuracy Graph:
ggplot(Lipify, aes(x = Category, y=Test.Accuracy)) + 
  geom_bar(stat = "identity",aes(fill = Test.Accuracy)) +
  ylim(0, 100) +
  ylab("Test Accuracy")+
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=16,face="bold"))


# Train Accuracy Graph:
ggplot(Lipify, aes(x = Category, y=Train.Accuracy)) + 
  geom_bar(stat = "identity",aes(fill = Train.Accuracy)) +
  ylim(0, 100) +
  ylab("Train Accuracy")+
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=16,face="bold"))


Lipify <- melt(Lipify, id.vars='Category')

ggplot(Lipify, aes(Category, value)) + geom_bar(aes(fill = variable), width = 0.4,
                                                position = position_dodge(width=0.5), stat="identity") + 
  theme(legend.position="top", legend.title = element_blank()) +
  ylab("Model Accuracy") +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=16,face="bold"))