pacman::p_load(tidyverse, readr)

# Vi indlæser dataene
behavior <- read_csv("data/behavior.csv")
cancellation <- read_csv("data/cancellation.csv")
subscription <- read_csv("data/subscription.csv")

glimpse(behavior)
glimpse(cancellation)
glimpse(subscription)

# Vi merger subscription og cancellation
joinet_data <- subscription %>%
  left_join(cancellation, by = "pseudo_id")

glimpse(joinet_data)

# Vi tjekker for dubletter
subscription_dub <- subscription %>%
  count(pseudo_id) %>%
  filter(n > 1)

cancellation_dub <- cancellation %>%
  count(pseudo_id) %>%
  filter(n > 1)

# Vi gør subscription til en række pr. bruger
subscription_clean <- subscription %>%
group_by(pseudo_id) %>%
  slice_max(order_date) %>%
  ungroup()

# Vi gør cancellation til en række pr. bruger
cancellation_clean <- cancellation %>%
  group_by(pseudo_id) %>%
  slice_max(expiration_date) %>%
  ungroup()

# Vi left_joiner / merger dem
sub_cancel <- subscription_clean %>%
  left_join(cancellation_clean, by = "pseudo_id")

# Vi beregner abonnementslængden
sub_cancel <- sub_cancel %>%
  mutate(
    end_date = if_else(
      is.na(subscription_cancel_date),
      Sys.Date(),
      subscription_cancel_date
    ),
    subscription_length_days = as.numeric(end_date - as.Date(order_date))
  )

# Vi laver grupper baseret på abonnementslængder
sub_cancel <- sub_cancel %>%
  mutate(
    length_group = case_when(
      subscription_length_days < 30 ~ "0–30 dage",
      subscription_length_days < 90 ~ "30–90 dage",
      subscription_length_days < 180 ~ "90–180 dage",
      subscription_length_days < 365 ~ "180–365 dage",
      TRUE ~ "365+ dage"
    )
  )

# Vi ser fordelingen
sub_cancel %>% count(length_group)

