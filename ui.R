library(shiny)

shinyUI(pageWithSidebar(

	headerPanel('Template'),
	
	sidebarPanel(
		
				
	),  ## close sidebarPanel
	
	mainPanel(
		tabsetPanel(		
			tabPanel(title='', 
				), ## close tabPanel 1
			
			tabPanel(title = '',
				)   ## Close tabPanel 2
			
			)	## close tabsetPanel
		)	## close mainPanel
	)	## close pageWithSidebar
)	## close shinyUI