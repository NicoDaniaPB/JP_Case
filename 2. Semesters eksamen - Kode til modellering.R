pacman::p_load(tidyverse, lubridate, readr)

# 1. Indlæsning af data -------------------------------------------------------

behavior <- read_csv("data/behavior.csv")
cancellation <- read_csv("data/cancellation.csv")
subscription2 <- read_csv("data/subscription_v2.csv")

# 1B. Split og parse subscription2 -------------------------------------------

subscription2_clean <- subscription2 %>%
  separate(
    col = 1,
    into = c(
      "pseudo_id", "account_active_days", "subscription_cancel_date",
      "order_date", "koen", "birthdate", "usr_created", "order_trackertag",
      "permission_given_order", "permission_given_today",
      "previous_subscriptions", "previous_campaigns", "previous_trials",
      "first_campaign_day", "last_campaign_day",
      "newsletters_before_order", "newsletters_after_order"
    ),
    sep = ";",
    convert = FALSE
  ) %>%
  mutate(
    # Datoer i dd-mm-yyyy format
    subscription_cancel_date = dmy(subscription_cancel_date),
    subscription_cancel_date = if_else(
      subscription_cancel_date == as.Date("3000-01-01"),
      NA_Date_,
      subscription_cancel_date
    ),
    birthdate = dmy(birthdate),
    usr_created = dmy(usr_created),
    first_campaign_day = dmy(first_campaign_day),
    last_campaign_day = dmy(last_campaign_day),
    
    # order_date er datetime
    order_date = ymd_hms(order_date),
    
    # Tal
    account_active_days = as.numeric(account_active_days),
    previous_subscriptions = as.numeric(previous_subscriptions),
    previous_campaigns = as.numeric(previous_campaigns),
    previous_trials = as.numeric(previous_trials),
    newsletters_before_order = as.numeric(newsletters_before_order),
    newsletters_after_order = as.numeric(newsletters_after_order),
    
    # Logiske værdier
    permission_given_order = as.logical(permission_given_order),
    permission_given_today = as.logical(permission_given_today), 
    
    # 27 ukendte køn 
    koen = if_else(koen == "" | is.na(koen), "Ukendt", koen)
  )

# 2. Fjern dubletter ---------------------------------------------------------

subscription_renset <- subscription2_clean %>%
  group_by(pseudo_id) %>%
  slice_max(order_date, with_ties = FALSE) %>%
  ungroup()

cancellation_renset <- cancellation %>%
  group_by(pseudo_id) %>%
  slice_max(expiration_date, with_ties = FALSE) %>%
  ungroup()


# 3. Merge subscription2 + cancellation ---------------------------------------

sub_cancel <- subscription_renset %>%
  left_join(cancellation_renset, by = "pseudo_id")


# 4. Beregning af abonnementslængde -------------------------------------------

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

sub_cancel <- sub_cancel %>%
  filter(!is.na(birthdate)) %>%
  filter(as.numeric(difftime(as.Date(order_date), birthdate, units = "days")) / 365 >= 18) %>%
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


# 7. Lav variabler til modeller ----------------------------------------

sub_cancel <- sub_cancel %>%
  mutate(
    continued_after_campaign = if_else(
      is.na(subscription_cancel_date) |
        subscription_cancel_date > last_campaign_day,
      1, 0
    ),
    fast_churn = if_else(
      subscription_length_days <= 3 & continued_after_campaign == 0,
      1, 0
    ),
    fast_churn_converted = if_else(
      subscription_length_days <= 3 & continued_after_campaign == 1,
      1, 0
    ),
    churn_10 = if_else(
      continued_after_campaign == 1 &
        !is.na(subscription_cancel_date) &
        as.numeric(subscription_cancel_date - last_campaign_day) <= 10,
      1, 0
    ),
    account_active_days_before_campaign = account_active_days -
      as.numeric(last_campaign_day - as.Date(order_date))
  )


# 8. Aggreger behavior-data --------------------------------------------------

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

full_data <- sub_cancel %>%
  left_join(behavior_features, by = "pseudo_id") %>%
  mutate(
    across(c(visits, unique_pages, restricted_views,
             restricted_ratio, avg_scroll, mobile_ratio,
             desktop_ratio), ~replace_na(., 0)),
    usr_created = if_else(is.na(usr_created), as.Date(order_date), usr_created),
    type   = if_else(is.na(type) & !is.na(subscription_cancel_date), "Ukendt", type),
    reason = if_else(is.na(reason) & !is.na(subscription_cancel_date), "Ukendt", reason),
    expiration_date = if_else(
      is.na(expiration_date) & !is.na(subscription_cancel_date),
      subscription_cancel_date,
      expiration_date
    )
  )


# 9. Variabler til klyngeanalyse  -----------------------------------------

mode_safe <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

behavior_agg <- behavior %>%
  mutate(
    er_artikel    = grepl("ECE\\d{8}", page_url_clean),
    er_restricted = page_restricted == "yes",
    dato          = as.Date(dt),
    sektion = case_when(
      grepl("/indland/", page_url_clean)          ~ "indland",
      grepl("/international/",  page_url_clean)   ~ "international",
      grepl("/sport/",   page_url_clean)          ~ "sport",
      grepl("/erhverv/", page_url_clean)          ~ "erhverv",
      grepl("/kultur/",  page_url_clean)          ~ "kultur",
      grepl("/opinion/", page_url_clean)          ~ "opinion",
      grepl("/debat/",   page_url_clean)          ~ "debat",
      page_url_clean == "https://jyllands-posten.dk/" ~ "forside",
      TRUE ~ "andet")) %>%
  group_by(pseudo_id) %>%
  summarise(
    antal_sidevisninger   = n(),
    antal_unikke_dage     = n_distinct(dato),
    antal_unikke_sider    = n_distinct(page_url_clean),
    antal_artikler        = sum(er_artikel),
    gns_scroll            = mean(scroll_depth, na.rm = TRUE),
    andel_restricted      = mean(er_restricted, na.rm = TRUE),
    er_mobil_primær       = as.integer(mode_safe(dvce_type) == "Mobile"),
    antal_devices         = n_distinct(dvce_type),
    andel_search          = mean(refr_medium == "search",   na.rm = TRUE),
    andel_internal        = mean(refr_medium == "internal", na.rm = TRUE),
    andel_email           = mean(refr_medium == "email",    na.rm = TRUE),
    andel_social          = mean(refr_medium == "social",   na.rm = TRUE),
    andel_indland         = mean(sektion == "indland"),
    andel_international   = mean(sektion == "international"),
    andel_sport           = mean(sektion == "sport"),
    andel_erhverv         = mean(sektion == "erhverv"),
    andel_forside         = mean(sektion == "forside"),
    gns_sider_pr_dag      = n() / n_distinct(dato),
    .groups = "drop"
  )

# 10. Gem det rensede datasæt -------------------------------------------------
saveRDS(behavior_agg, "data/datasæt_konstruerede_variabler.rds")
saveRDS(full_data, "data/renset_datasæt.rds")
full_data %>%
  mutate(across(where(is.Date), ~ as.character(.x))) %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), "", .x))) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4))) %>%
  write.table("data/renset_data.csv",
              sep = ";", dec = ",", row.names = FALSE, quote = FALSE)

#TEST KUN EN TEST EN FUCKING TEST
behavior_agg

behavior$page_url_clean

behavior %>% 
  filter(grepl("/erhverv/", page_url_clean)) %>% 
  count()

behavior %>% 
  filter(grepl("erhverv", page_url_clean)) %>% 
  count(page_url_clean) %>% 
  arrange(desc(n)) %>% 
  head(20)


behavior %>% 
  count(page_url_clean) %>% 
  arrange(desc(n)) %>% 
  head(50)
