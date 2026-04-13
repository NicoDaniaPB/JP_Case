#> pacman::p_load indlæser og installerer automatisk de nødvendige pakker.
#> shiny giver os værktøjerne til at bygge web apps i R.
#> bslib giver os moderne Bootstrap baserede UI komponenter,
#> som navbarPage der bruges til at bygge navigation og layout.
pacman::p_load("shiny", "bslib", "tidyverse")

# Indlæser data
model_data <- readRDS("renset_datasæt.rds")

#> ui: Definerer appens visuelle grænseflade (hvad brugeren ser)
#> navbarPage opretter en navbar med faner øverst.
#> title placerer appens navn til venstre i navbaren.
#> Logo placeres til højre i navbaren.
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
  tabPanel("Churn Prognose",
           sidebarLayout(
             sidebarPanel(
               numericInput("alder", "Alder:", value = NA, min = 18, max = 100),
               selectInput("kon", "Køn:", choices = c("Vælg køn" = "", "Mand", "Kvinde", "Andet")),
               numericInput("anciennitet", "Anciennitet (måneder):", value = NA, min = 0),
               numericInput("maanedlig_pris", "Månedlig pris (kr.):", value = NA, min = 0),
               selectInput("kontrakt", "Kontrakttype:",
                           choices = c("Vælg kontrakttype" = "", "Måned-til-måned", "1 år", "2 år")),
               selectInput("betaling", "Betalingsmetode:",
                           choices = c("Vælg betalingsmetode" = "", "Kreditkort", "Bankoverførsel", "MobilePay")),
               actionButton("predict", "Forudsig churn", class = "btn btn-primary w-100")
             ),
             mainPanel(
               h4("Resultat"),
               textOutput("churn_resultat")
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
  
  #> renderText genererer en tekstoutput som vises i mainPanel.
  #> input$predict registrerer klik på "Forudsig churn" knappen.
  #> Returner en placeholder tekst indtil modellen er tilsluttet.
  output$churn_resultat <- renderText({
    input$predict
    "Model ikke tilsluttet endnu - kommer senere."
  })
}

#> shinyApp sammensætter UI og server til en færdig applikation.
shinyApp(ui = ui, server = server)