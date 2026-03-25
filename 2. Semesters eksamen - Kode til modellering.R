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
      subscription_length_days < 270 ~ "180–270 dage",
      subscription_length_days < 365 ~ "270–365 dage",
      TRUE ~ "365+ dage"
    )
  )


# Vi ser fordelingen
sub_cancel %>% count(length_group)


# Vi tilføjer alder som variabel 
sub_cancel <- sub_cancel %>%
  mutate(
    age = floor(time_length(interval(birthdate, today()), "years"))
  )

# Vi beregner gennemsnitsalderen for de forskellige intervaller
sub_cancel %>%
  group_by(length_group) %>%
  summarise(
    mean_age = mean(age, na.rm = TRUE),
    median_age = median(age, na.rm = TRUE),
    n = n()
  )

# Vi laver et boxplot og aldersfordeling pr. abonnementslængnde
sub_cancel %>%
  ggplot(aes(x = length_group, y = age)) +
  geom_boxplot(fill = "steelblue", alpha = 0.6) +
  labs(
    title = "Aldersfordeling pr. abonnementslængde",
    x = "Abonnementslængde",
    y = "Alder"
  ) +
  theme_minimal()

# Vi laver nogle forskellige demografiske grupper 
sub_cancel <- sub_cancel %>%
  mutate(
    age_group = case_when(
      age < 25 ~ "18–24",
      age < 35 ~ "25–34",
      age < 45 ~ "35–44",
      age < 55 ~ "45–54",
      age < 65 ~ "55–64",
      TRUE ~ "65+"
    )
  )

# Vi ser fordelingen af aldersgrupper indenfor abonnementslængden
sub_cancel %>%
  count(length_group, age_group) %>%
  group_by(length_group) %>%
  mutate(pct = n / sum(n) * 100)

view(sub_cancel)

# Vi laver et stacked bar chat
sub_cancel %>%
  ggplot(aes(x = length_group, fill = age_group)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Aldersgrupper fordelt på abonnementslængde",
    x = "Abonnementslængde",
    y = "Andel",
    fill = "Aldersgruppe"
  ) +
  theme_minimal()




