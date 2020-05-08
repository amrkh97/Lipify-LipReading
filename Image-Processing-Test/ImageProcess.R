rm(list=ls())
#display work items
ls()
#get working directory
getwd()
library(ggplot2)
df <- read.csv('ModelsTiming.csv')
dfm <- df[,3:4]
# accuracy <- df[df$Video.Name == "Colors_7.mp4", ] 
levels(df$Model) <- c(levels(df$Model), "Blue") 
df$Model[df$Model == "Dlib"]  <- "Blue" 
dfm <- as.matrix(dfm)

barplot(dfm,
        main = "Models Preformance",
        col = df$Model,
        beside = TRUE
)
label<- unique(df$Model)
legend("topright",
       c("Dlib", "HaarCascade", "SkinDetection"),
       fill = label
)

