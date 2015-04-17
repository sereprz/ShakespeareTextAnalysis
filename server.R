library(shiny)
require(ggplot2)
require(reshape)

col <- c('darkorchid4','darkorange', 'darkred', 'cornflowerblue', 'darkblue', 'darkgoldenrod1', 'darkcyan')
titles <- read.csv('titles.csv', header =FALSE)
names(titles) <- c('play','label','year','cluster')
doc_topic <- read.csv('doc_topic.csv', header = FALSE)
names(doc_topic) <- paste('topic', 0:11, sep ='')
doc_topic$group <- clusters
doc_topic$name <- titles$play
m_doc_topic <- melt(doc_topic, id = c(13,14))
centroids <- rbind(apply(topic_group0, 2, mean), apply(topic_group1, 2, mean))

tmp <- with(m_doc_topic, aggregate(value, list(variable, group), mean))

shinyServer(function(input, output) {

    ggplot(titles, aes(x = factor(cluster), y = year, group = factor(cluster), fill = factor(cluster))) + geom_boxplot(notch = TRUE, notchwidth = 0.2) + scale_fill_manual(values = col) + guides(fill = FALSE) + theme_bw()

    ggplot(tmp, aes(factor(Group.2), x, fill = Group.1, group = Group.1)) + geom_bar(width = .4, stat='identity', position = 'stack') + scale_fill_brewer(palette = 'Paired') + theme_bw() + coord_flip() + ylab('Probability') + xlab('Cluster') + guides(fill=guide_legend(title=NULL, size = 1))

    ggplot(titles, aes(x = year, y = 1, color = factor(cluster))) + geom_point(size = 8, shape = '*') + scale_color_manual(name = 'Cluster', values = col) + theme_bw() + xlab('Year') + ylab('') + theme(axis.ticks = element_blank(), axis.text.y = element_blank())

}) # close shinyServer
				