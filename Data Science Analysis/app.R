library(shiny)
library(shinycssloaders)
library(wordcloud)
library(ggplot2)
library(shinydashboard)
library(dplyr)
library(tidytext)
library(DT)

source("scrapper/review.R")
source("classifier/naive_bayes.R")


features <- readRDS(features_rds_path)


ui <- dashboardPage(
  
  dashboardHeader(title = "Tripadvisor Restaurant Review"),
  
  dashboardSidebar(
    
    
    sliderInput(
      "size",
      "Total reviews",
      min = 0,
      max = 1000,
      value = 10
      
      
    ),
    ui <- fluidPage(
      # All your styles will go here
      tags$style(HTML(".js-irs-0 .irs-single, .js-irs-0 .irs-bar-edge, .js-irs-0 .irs-bar {background-color: #262626}")),
    ),
    fluidPage(
      submitButton("Submit"),
    ),
    
    
    hr(),
    helpText(
      "Review Restaurant yang di-scrape akan di klasifikasikan dengan Naive Bayes"
    ),
    hr(),
    helpText(
      "Peringatan: Mungkin terjadi lost connection saat scraping data. Refresh halaman jika terjadi error.", style = "color:#d9534f"
    )
  ),
  
  dashboardBody(
    tags$style(type='text/css', ".skin-blue .main-header .logo {background-color: #262626}" ),
    tags$style(type='text/css', ".skin-blue .main-header .logo:hover {background-color: #262626}"),
    tags$style(type='text/css', ".skin-blue .main-header .navbar {background-color: #262626}"),
    tags$style(type="text/css",".shiny-output-error { visibility: hidden; }",".shiny-output-error:before { visibility: hidden; }"),
    
    textInput(
      width = "1500px",
      "url",
      "Enter tripadvisor restauran url", 
      placeholder = "url", 
      value = "https://www.tripadvisor.com/Restaurant_Review-g37209-d2018616-Reviews-High_Velocity-Indianapolis_Indiana.html"
    ),
    fluidRow(
      valueBoxOutput("jmlh_rev"),
      valueBoxOutput("satisfied"),
      valueBoxOutput("unsatisfied")
    ),
    
    fluidRow(
      tags$style(HTML('table.dataTable thead tr td {color: black !important;}')),
      tags$style(HTML(".dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing,.dataTables_wrapper .dataTables_paginate .paginate_button, .dataTables_wrapper .dataTables_paginate .paginate_button.disabled {
            color: #303030 !important;
        }")),
      box(
        title = "Sentiment Analysis",
        
        solidHeader = T,
        width = 12,
        collapsible = T,
        div(DT::dataTableOutput("table_review") %>% withSpinner(color="#1167b1"), style = "font-size: 90%;")
      ),
    ),
    fluidRow(
      box(title = "Word Cloud",
          
          solidHeader = T,
          width = 6,
          collapsible = T,
          plotOutput("wordcloud") %>% withSpinner(color="#1167b1")
      ),
      box(title = "WordCount",
          
          solidHeader = T,
          width = 6,
          collapsible = T,
          plotOutput("word_count") %>% withSpinner(color="#1167b1")
      ),
      fluidRow(
        box(title = "Sentiment Positive vs Negative",
            solidHeader = T,
            width = 12,
            collapsible = T,
            plotOutput("sentiment_contribution") %>% withSpinner(color="#1167b1")
        )
      )
    )
  )
)


server <- function(input, output) {
  
  data <- reactive({
    withProgress({
      setProgress(message = "Collecting data", value = 0)
      
      result <- get_restaurant_reviews(input$url, input$size, incProgress)
    })
    
    return(result)
  })
  
  prediction_data <- reactive({
    withProgress({
      setProgress(message = "Predicting sentiment", value = 0)
      
      ulasan <- data()$review
      incProgress(1/2)
      prediction <- predict_sentiment(ulasan)
      incProgress(1/2)
    })
    prediction$reviewer <- data()$reviewer
    return(prediction)
  })
  
  create_dtm <- function(data) {
    corpus <- Corpus(VectorSource(data))
    
    corpus_clean <- corpus %>%
      tm_map(content_transformer(tolower)) %>% 
      tm_map(removePunctuation) %>%
      tm_map(removeNumbers) %>%
      tm_map(removeWords, stopwords(kind="en")) %>%
      tm_map(stripWhitespace)
    
    create_dtm <- corpus_clean %>%
      DocumentTermMatrix(control=list(dictionary = features))
  }
  
  dataword <- reactive({
    v <- sort(colSums(as.matrix(create_dtm(data()$review))), decreasing = TRUE)
    data.frame(Kata=names(v), Jumlah=as.integer(v), row.names=NULL, stringsAsFactors = FALSE) %>%
      filter(Jumlah > 0)
  })
  
  output$jmlh_rev <- renderValueBox({
    valueBox(
      "Total", 
      paste0(nrow(prediction_data()), " review"),
      icon = icon("poll-h", class = "fas fa-poll-h", lib = "font-awesome"),
      color = "blue"
      )
    
  })
  
  
  output$satisfied <- renderValueBox({
    valueBox(
      "statisfied", 
      paste0(nrow(prediction_data() %>% filter(sentiment == "Positive")), " review"),
      icon = icon("smile"),
      color = "green")
  })
  
  output$unsatisfied <- renderValueBox({
    valueBox(
      "unsatisfied",
      paste0(nrow(prediction_data() %>% filter(sentiment == "Negative")), " review"), 
      icon = icon("frown"),
      color = "red")
    })
  
  # plot sentiment positive vs negative
  output$sentiment_contribution <- renderPlot({
    sentiments <- dataword() %>% 
      inner_join(get_sentiments("bing"), by = c("Kata" = "word"))
    
    positive <- sentiments %>% filter(sentiment == "positive") %>% top_n(10, Jumlah) 
    negative <- sentiments %>% filter(sentiment == "negative") %>% top_n(10, Jumlah)
    sentiments <- rbind(positive, negative)
    
    sentiments <- sentiments %>%
      mutate(Jumlah=ifelse(sentiment =="negative", -Jumlah, Jumlah))%>%
      mutate(Kata = reorder(Kata, Jumlah))
    
    ggplot(sentiments, aes(Kata, Jumlah, fill=sentiment))+
      geom_bar(stat = "identity")+scale_fill_manual(values = c("#a8a8a8", "#262626"))+
      theme(axis.text.x = element_text(angle = 90, hjust = 1))+
      ylab("Sentiment Contribution") + xlab("Word")
  })
  
  output$table_review <- renderDataTable(datatable({
    prediction_data()
  }, extensions = 'Buttons', options = list(dom = 'Bfrtip')))
  
  
  output$wordcloud <- renderPlot({
    data.corpus <- clean_data(data()$review)
    wordcloud(data.corpus, min.freq = 30, max.words = 50)
  })
  
  output$word_count <- renderPlot({
    countedWord <- dataword() %>%
      top_n(10, Jumlah) %>%
      mutate(Kata = reorder(Kata, Jumlah))
    
    ggplot(countedWord, aes(Kata, Jumlah, fill = Jumlah)) + scale_fill_gradient(low="#a8a8a8", high="#262626")+
      geom_col() +
      guides(fill = FALSE) +
      theme_minimal()+
      labs(x = NULL, y = "Word Count") +
      ggtitle("Most Frequent Words") +
      coord_flip()
  })
}

shinyApp(ui, server)