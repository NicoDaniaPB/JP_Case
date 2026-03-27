pacman::p_load(tidyverse, lubridate, readr)

# 1. Indlæsning af data -------------------------------------------------------

# Vi indlæser de tre datasæt via read_csv-funktionen.

behavior <- read_csv("data/behavior.csv")
cancellation <- read_csv("data/cancellation.csv")
subscription <- read_csv("data/subscription.csv")

# Behavior beskriver kundernes digitale adfærd.
# Cancellation beskriver opsigelser og årsagerne hertil.


# 2. Fjern dubletter ---------------------------------------------------------

# Vi fjerner dubeletter i datasættet, da det kan give problemer i vores 
# data-analyse. Det kan give problemer som dobbelt tælling af churn, forkerte 
# churb-targets, overfitting i modellerne eler skæve segmenter i klyngeanalysen.
# Vi vælger derfor den nyeste ordre pr. kunde, og den seneste opsigelse 
# pr. kunde.

subscription_renset <- subscription %>%
  group_by(pseudo_id) %>%
  slice_max(order_date) %>%
  ungroup()

cancellation_renset <- cancellation %>%
  group_by(pseudo_id) %>%
  slice_max(expiration_date) %>%
  ungroup()


# 3. Merge subscription + cancellation ---------------------------------------

# Vi kombinerer de to rensede datasæt, så vi har et samlet datasæt pr. kunde. 

sub_cancel <- subscription_renset %>%
  left_join(cancellation_renset, by = "pseudo_id")


# 4. Beregning af abonnementslængde -------------------------------------------

# Vi laver intervaller for abonnementslængden for kunderne. Dette er et 
# fundament for churn-analysen og segmenteringen. 

sub_cancel <- sub_cancel %>%
  mutate(
    end_date = if_else(
      is.na(subscription_cancel_date),
      Sys.Date(),
      subscription_cancel_date
    ),
    subscription_length_days = as.numeric(end_date - as.Date(order_date))
  )


# 5. Lav churn-grupper -------------------------------------------------------

# Vi opdeler kunder i kategorier som baseres på, hvor langt tid deres 
# abonnement har aktiv. Dette skal bruges til segmentering og visalusiering, 
# samt forståelse af churn-mønstre. 

sub_cancel <- sub_cancel %>%
  mutate(
    length_group = case_when(
      subscription_length_days < 30 ~ "0–30 dage",
      subscription_length_days < 90 ~ "30–90 dage",
      subscription_length_days < 180 ~ "90–180 dage",
      subscription_length_days < 270 ~ "180–270 dage",
      subscription_length_days < 365 ~ "270–365 dage",
      TRUE ~ "365+ dage"
    )
  )


# 6. Tilføj alder + aldersgrupper -------------------------------------------

# Vi tilføjer to nye variabler: alder og aldersgrupper.
# Vi beregner alder ud fra fødselsdato, og kategoriser dem i breddere grupper.
# Dette skal bruges til churn-analyse, segmentering og ML-modellerne. 

sub_cancel <- sub_cancel %>%
  mutate(
    age = floor(time_length(interval(birthdate, today()), "years")),
    age_group = case_when(
      age < 25 ~ "18–24",
      age < 35 ~ "25–34",
      age < 45 ~ "35–44",
      age < 55 ~ "45–54",
      age < 65 ~ "55–64",
      TRUE ~ "65+"
    )
  )

# 7. Lav churn-variabler til modeller ----------------------------------------

# Vi laver churn-variabler, som vi skal bruge senere til ML-modellerne.

# Første ML-target ser ud som følgende: 
# Hvis kunden stadig er aktiv eller opsiger efter efter kampagnen -> 1
# Hvis kunden opsiger inden kampagnen slutter -> 0

# Model 1: Churn ved kampagneslut
sub_cancel <- sub_cancel %>%
  mutate(
    continued_after_campaign = if_else(
      is.na(subscription_cancel_date) |
        subscription_cancel_date > last_campaign_day,
      1, 0
    )
  )

# Vores andet ML-target skal tjek til hurtig churn (< 90). 
# Hvis skunden chruner inden for 90 dage -> 1
# Ellers er den -> 0

# Model 2: Hurtig churn (< 90 dage)
sub_cancel <- sub_cancel %>%
  mutate(
    fast_churn = if_else(subscription_length_days < 90, 1, 0)
  )


# 8. Aggreger behavior-data --------------------------------------------------

# Vi reducerer behavior-dataene til en række pr. kunde ved at beregne: 
# antal besøg
# antal unikke sider
# antal artikler bag login
# hvor stor en andel af siderne var restricted
# gennemsnitlig scroll‑dybde
# device‑fordeling

# Dette er værdiefulde features til vores churn-modeller. 

behavior_features <- behavior %>%
  group_by(pseudo_id) %>%
  summarise(
    visits = n(),
    unique_pages = n_distinct(page_url_clean),
    restricted_views = sum(page_restricted == "yes", na.rm = TRUE),
    restricted_ratio = restricted_views / visits,
    avg_scroll = mean(scroll_depth, na.rm = TRUE),
    mobile_ratio = mean(dvce_type == "Mobile"),
    desktop_ratio = mean(dvce_type == "Computer")
  )

# Vi merger behavior ind i datasættet
full_data <- sub_cancel %>%
  left_join(behavior_features, by = "pseudo_id")


# 9. Gem det rensede datasæt -------------------------------------------------

# Vi gemmer modellen som RDS-fil, så vi kan bruge den til modelleringskoden
saveRDS(full_data, "data/renset_datasæt.rds")

