pacman::p_load("shiny", "bslib", "tidyverse", "xgboost")

#> pacman::p_load indlæser og installerer automatisk de nødvendige pakker.
#> shiny giver os værktøjerne til at bygge web apps i R.
#> bslib giver os moderne Bootstrap baserede UI komponenter.
#> xgboost bruges til at køre prognosen med den gemte model.

# Indlæser data og model
model_data <- readRDS("renset_datasæt.rds")
xgb_model2 <- readRDS("xgb_model2.rds")

# Beregn gennemsnit til default-værdier for adfærdsvariablerne
means <- model_data %>%
  filter(continued_after_campaign == 1) %>%
  summarise(
    across(c(visits, unique_pages, restricted_views, restricted_ratio,
             avg_scroll, mobile_ratio, desktop_ratio, account_active_days),
           ~ round(mean(.x, na.rm = TRUE), 2))
  )

mean_visits               <- as.numeric(means$visits)
mean_unique_pages         <- as.numeric(means$unique_pages)
mean_restricted_views     <- as.numeric(means$restricted_views)
mean_restricted_ratio     <- as.numeric(means$restricted_ratio)
mean_avg_scroll           <- as.numeric(means$avg_scroll)
mean_mobile_ratio         <- as.numeric(means$mobile_ratio)
mean_desktop_ratio        <- as.numeric(means$desktop_ratio)
mean_account_active_days  <- as.numeric(means$account_active_days)

count_value <- function(val) {
  if (val == "5+") 5 else as.numeric(val)
}

#> ui: Definerer appens visuelle grænseflade (hvad brugeren ser)
ui <- navbarPage(
  title = div(
    "Jyllands-Posten – Kampagneanalyse",
    tags$div(
      style = "position: absolute; right: 20px; top: 50%; transform: translateY(-50%); display: flex; align-items: center; gap: 10px;",
      tags$img(src = 'nim_logo_hvid.png', height = '60px')
    )
  ),
  
  header = tagList(
    tags$style(HTML("
    .navbar { background-color: #1a1a1a !important; margin-bottom: 0 !important; }
    .navbar-brand { color: white !important; font-size: 24px !important; font-weight: bold !important; }
    .navbar-nav > li > a { color: #aaaaaa !important; font-weight: 500; }
    .navbar-nav > li > a:hover { color: white !important; background-color: #333333 !important; }
    .navbar-nav > .active > a,
    .navbar-nav > .active > a:hover,
    .navbar-nav > .active > a:focus { background-color: #333333 !important; color: white !important; border-bottom: 3px solid white !important; }
    .container-fluid { padding-top: 0 !important; padding-left: 0 !important; padding-right: 0 !important; }
    body { padding-top: 0 !important; }
    .tab-pane { padding: 15px !important; }
  "))
  ),
  
  tabPanel("Kampagneoverblik",
           fluidRow(
             column(4,
                    div(style = "text-align: center; background-color: #1a1a1a; color: white; border-radius: 8px; padding: 20px;",
                        h2(textOutput("antal_abonnenter"), style = "margin: 0; font-size: 36px;"),
                        p("Deltagere i kampagnen", style = "margin: 5px 0 0; color: #aaaaaa;")
                    )
             ),
             column(4,
                    div(style = "text-align: center; background-color: #2980B9; color: white; border-radius: 8px; padding: 20px;",
                        h2(textOutput("andel_fortsatte"), style = "margin: 0; font-size: 36px;"),
                        p("Fortsatte abonnementet", style = "margin: 5px 0 0; color: #d0e8f7;")
                    )
             ),
             column(4,
                    div(style = "text-align: center; background-color: #E8870A; color: white; border-radius: 8px; padding: 20px;",
                        h2(textOutput("andel_churn"), style = "margin: 0; font-size: 36px;"),
                        p("Fortsatte ikke abonnementet", style = "margin: 5px 0 0; color: #fce8cc;")
                    )
             )
           ),
           hr(),
           h4("Udfald efter kampagne"),
           plotOutput("plot_continued")
  ),
  
  tabPanel("Model 1 – Churn efter kampagne",
           sidebarLayout(
             sidebarPanel(
               p("Model 1 kommer senere.", style = "color: #aaaaaa;")
             ),
             mainPanel(
               h4("Resultat"),
               p("Model ikke tilsluttet endnu.")
             )
           )
  ),
  
  tabPanel("Model 2 – Churn efter konvertering",
           fluidRow(
             column(4,
                    h5("Kundeoplysninger"),
                    
                    selectInput("m2_koen", "Køn:",
                                choices = c("Kvinde", "Mand", "Ukendt"),
                                selected = "Mand"),
                    
                    numericInput("m2_age", "Alder:",
                                 value = 55, min = 18, max = 100),
                    
                    numericInput("m2_account_active_days", "Aktive dage hos JP (historik):",
                                 value = mean_account_active_days, min = 0),
                    
                    selectInput("m2_previous_subscriptions", "Tidligere abonnementer:",
                                choices = c("0", "1", "2", "3", "4", "5+"),
                                selected = "2"),
                    
                    selectInput("m2_previous_campaigns", "Tidligere kampagner:",
                                choices = c("0", "1", "2", "3", "4", "5+"),
                                selected = "1"),
                    
                    selectInput("m2_previous_trials", "Tidligere prøveabonnementer:",
                                choices = c("0", "1", "2", "3", "4", "5+"),
                                selected = "1"),
                    
                    checkboxInput("m2_permission", "Samtykke til markedsføring ved køb", value = FALSE),
                    
                    selectInput("m2_newsletters_before", "Nyhedsbreve før ordre:",
                                choices = c("0", "1", "2", "3", "4", "5+"),
                                selected = "0"),
                    
                    selectInput("m2_newsletters_after", "Nyhedsbreve efter ordre:",
                                choices = c("0", "1", "2", "3", "4", "5+"),
                                selected = "0")
             ),
             
             column(4,
                    h5("Adfærd (første 30 dage)"),
                    
                    numericInput("m2_visits", "Antal besøg:",
                                 value = mean_visits, min = 0),
                    
                    numericInput("m2_unique_pages", "Unikke sider:",
                                 value = mean_unique_pages, min = 0),
                    
                    numericInput("m2_restricted_views", "Låste artikler set:",
                                 value = mean_restricted_views, min = 0),
                    
                    sliderInput("m2_restricted_ratio", "Andel låst indhold:",
                                min = 0, max = 1, value = mean_restricted_ratio, step = 0.01),
                    
                    sliderInput("m2_avg_scroll", "Gennemsnitlig scroll:",
                                min = 0, max = 1, value = mean_avg_scroll, step = 0.01),
                    
                    sliderInput("m2_mobile_ratio", "Mobilandel:",
                                min = 0, max = 1, value = mean_mobile_ratio, step = 0.01),
                    
                    sliderInput("m2_desktop_ratio", "Desktopandel:",
                                min = 0, max = 1, value = mean_desktop_ratio, step = 0.01),
                    
                    br(),
                    actionButton("m2_predict", "Forudsig churn", class = "btn btn-primary w-100")
             ),
             
             column(4,
                    h5("Resultat"),
                    br(),
                    uiOutput("m2_resultat")
             )
           )
  ),
  
  tabPanel("Om os",
           h4("Om os"),
           p("Indhold kommer senere.")
  )
)

#> server definerer appens logik, altså hvad der sker bag kulisserne når brugeren interagerer.
#> input indeholder værdier fra UI. F.eks. hvad brugeren har indtastet.
#> output indeholder det der sendes tilbage til UI. F.eks. tekst eller plots.
server <- function(input, output, session) {
  
  output$antal_abonnenter <- renderText({
    format(nrow(model_data), big.mark = ".", decimal.mark = ",")
  })
  
  output$andel_fortsatte <- renderText({
    andel <- mean(model_data$continued_after_campaign, na.rm = TRUE) * 100
    paste0(round(andel, 1), "%")
  })
  
  #> Beregner andelen der churnede efter kampagnen.
  #> continued_after_campaign == 0 betyder at kunden ikke fortsatte.
  output$andel_churn <- renderText({
    andel <- mean(model_data$continued_after_campaign == 0, na.rm = TRUE) * 100
    paste0(round(andel, 1), "%")
  })
  
  #> Laver et søjlediagram over fordelingen af continued_after_campaign.
  #> 0 = churnede efter kampagne, 1 = fortsatte med abonnement.
  #> Farverne blå og orange er gennemgående i alle visualiseringer.
  output$plot_continued <- renderPlot({
    model_data |>
      mutate(
        status = ifelse(continued_after_campaign == 1, "Fortsatte", "Churnede"),
        status = fct_relevel(status, "Fortsatte", "Churnede")
      ) |>
      count(status) |>
      mutate(pct = n / sum(n) * 100) |>
      ggplot(aes(x = status, y = n, fill = status)) +
      geom_col(width = 0.5) +
      geom_text(aes(label = paste0(n, " (", round(pct, 1), "%)")),
                vjust = -0.5, size = 5.5, fontface = "bold") +
      scale_fill_manual(values = c("Fortsatte" = "#2980B9",
                                   "Churnede"  = "#E8870A")) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
      labs(x = NULL, y = "Antal abonnenter") +
      theme_minimal(base_size = 16) +
      theme(legend.position = "none",
            panel.grid.major.x = element_blank(),
            axis.text = element_text(size = 13),
            axis.title = element_text(size = 14))
  })
  
  #> Model 2 prognose
  #> Når brugeren klikker på "Forudsig churn" beregnes sandsynligheden
  #> for at kunden churner inden for 10 dage efter kampagnens afslutning.
  observeEvent(input$m2_predict, {
    
    prev_subs   <- count_value(input$m2_previous_subscriptions)
    prev_camp   <- count_value(input$m2_previous_campaigns)
    prev_trials <- count_value(input$m2_previous_trials)
    news_before <- count_value(input$m2_newsletters_before)
    news_after  <- count_value(input$m2_newsletters_after)
    
    age_at_order               <- input$m2_age
    days_since_user_created    <- 0
    total_previous_engagement  <- prev_subs + prev_camp + prev_trials
    has_previous_subscriptions <- ifelse(prev_subs > 0, 1, 0)
    
    new_data <- data.frame(
      koenKvinde                 = ifelse(input$m2_koen == "Kvinde", 1, 0),
      koenMand                   = ifelse(input$m2_koen == "Mand", 1, 0),
      koenUkendt                 = ifelse(input$m2_koen == "Ukendt", 1, 0),
      age                        = input$m2_age,
      previous_subscriptions     = prev_subs,
      previous_campaigns         = prev_camp,
      previous_trials            = prev_trials,
      account_active_days        = input$m2_account_active_days,
      permission_given_orderTRUE = ifelse(input$m2_permission, 1, 0),
      newsletters_before_order   = news_before,
      newsletters_after_order    = news_after,
      visits                     = input$m2_visits,
      unique_pages               = input$m2_unique_pages,
      restricted_views           = input$m2_restricted_views,
      restricted_ratio           = input$m2_restricted_ratio,
      avg_scroll                 = input$m2_avg_scroll,
      mobile_ratio               = input$m2_mobile_ratio,
      desktop_ratio              = input$m2_desktop_ratio,
      days_since_user_created    = days_since_user_created,
      age_at_order               = age_at_order,
      total_previous_engagement  = total_previous_engagement,
      has_previous_subscriptions = has_previous_subscriptions
    )
    
    input_matrix <- as.matrix(new_data)
    prob <- predict(xgb_model2, xgb.DMatrix(input_matrix))
    
    output$m2_resultat <- renderUI({
      farve <- if (prob >= 0.5) "#E8870A" else "#2980B9"
      vurdering <- if (prob >= 0.5) "Høj risiko for churn" else "Lav risiko for churn"
      tagList(
        div(style = paste0("text-align: center; background-color: ", farve,
                           "; color: white; border-radius: 8px; padding: 30px;"),
            h2(paste0(round(prob * 100, 1), "%"), style = "margin: 0; font-size: 48px;"),
            p("Sandsynlighed for churn inden for 10 dage", style = "margin: 5px 0 0;"),
            hr(style = "border-color: rgba(255,255,255,0.3);"),
            p(strong(vurdering), style = "margin: 0;")
        )
      )
    })
  })
}

#> shinyApp sammensætter UI og server til en færdig applikation.
shinyApp(ui = ui, server = server)
