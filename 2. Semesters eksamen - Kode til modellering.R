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

# Se de rækker der ville blive fjernet
subscription2_clean %>%
  group_by(pseudo_id) %>%
  filter(n() > 1) %>%
  arrange(pseudo_id, order_date) %>%
  select(pseudo_id, order_date, subscription_cancel_date)
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
    churn_30 = if_else(
      continued_after_campaign == 1 &
        !is.na(subscription_cancel_date) &
        as.numeric(subscription_cancel_date - last_campaign_day) <= 30,
      1, 0
    )
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


# 9. Gem det rensede datasæt -------------------------------------------------

saveRDS(full_data, "data/renset_datasæt.rds")