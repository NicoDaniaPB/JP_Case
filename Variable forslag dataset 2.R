library(tidyverse)

behavior     <- read_csv("data/behavior.csv")
cancellation <- read_csv("data/cancellation.csv")
subscription <- read_csv("data/subscription.csv")

mode_safe <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

behavior_agg <- behavior %>%
  mutate(
    er_artikel  = grepl("ECE\\d{8}", page_url_clean),
    er_restricted = page_restricted == "yes",
    dato        = as.Date(dt),
    sektion = case_when(
      grepl("/indland/",  page_url_clean) ~ "indland",
      grepl("/udland/",   page_url_clean) ~ "udland",
      grepl("/sport/",    page_url_clean) ~ "sport",
      grepl("/okonomi/",  page_url_clean) ~ "oekonomi",
      grepl("/kultur/",   page_url_clean) ~ "kultur",
      grepl("/opinion/",  page_url_clean) ~ "opinion",
      grepl("/debat/",    page_url_clean) ~ "debat",
      page_url_clean == "https://jyllands-posten.dk/" ~ "forside",
      TRUE ~ "andet"
    )
  ) %>%
  group_by(pseudo_id) %>%
  summarise(
    # --- VOLUMEN ---
    antal_sidevisninger   = n(),
    antal_unikke_dage     = n_distinct(dato),
    antal_unikke_sider    = n_distinct(page_url_clean),
    antal_artikler        = sum(er_artikel),
    antal_unikke_artikler = n_distinct(page_url_clean[er_artikel]),
    
    # --- ENGAGEMENT ---
    andel_restricted  = mean(er_restricted, na.rm = TRUE),
    antal_restricted  = sum(er_restricted,  na.rm = TRUE),
    gns_scroll        = mean(scroll_depth,  na.rm = TRUE),
    andel_fuld_scroll = mean(scroll_depth >= 1.0, na.rm = TRUE),
    andel_ingen_scroll= mean(scroll_depth == 0.0, na.rm = TRUE),
    
    # --- DEVICE & OS ---
    primær_device  = mode_safe(dvce_type),
    er_mobil_primær= as.integer(mode_safe(dvce_type) == "Mobile"),
    antal_devices  = n_distinct(dvce_type),
    
    # --- TRAFIKKILDE ---
    andel_search   = mean(refr_medium == "search",   na.rm = TRUE),
    andel_internal = mean(refr_medium == "internal", na.rm = TRUE),
    andel_email    = mean(refr_medium == "email",    na.rm = TRUE),
    andel_social   = mean(refr_medium == "social",   na.rm = TRUE),
    primær_kanal   = mode_safe(refr_medium),
    
    # --- INDHOLDSPRÆFERENCE ---
    andel_indland  = mean(sektion == "indland"),
    andel_udland   = mean(sektion == "udland"),
    andel_sport    = mean(sektion == "sport"),
    andel_oekonomi = mean(sektion == "oekonomi"),
    andel_forside  = mean(sektion == "forside"),
    
    # --- ADFÆRDSMØNSTER OVER TID ---
    første_besøg      = min(dato),
    sidste_besøg      = max(dato),
    dage_aktiv_span   = as.integer(max(dato) - min(dato)),
    gns_sider_pr_dag  = n() / n_distinct(dato),
    
    .groups = "drop"
  )


write_csv(behavior_agg, "data/datasæt 2 - konstruerede variabler.csv")


